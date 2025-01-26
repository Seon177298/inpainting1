import torch
import cv2
import textwrap
import lpips
import brisque
import pyiqa
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from transformers import SamModel, SamProcessor
from PIL import Image
from diffusers import DDIMScheduler
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
import torchvision.transforms as T
import torch.nn.functional as F
from safetensors.torch import load_file
import os


def create_mask(image_path, output_path=None, threshold=128, use_otsu =False, inver_mask=False, normalize=False):
    # Check if image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at: {image_path}")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if use_otsu:
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        _, mask = cv2.threshold(gray_image, 0, 255, thresh_type)
    else:
        _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)

    if inver_mask:
        mask = cv2.bitwise_not(mask)

    if normalize:
        mask = (mask // 255).astype(np.uint8)

    # Apply threshold
    _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)

    # Save output if path provided
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, mask)

    return mask


def create_canny_mask(
    image_path, output_path=None, low_threshold=100, high_threshold=200
):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_image, low_threshold, high_threshold)

    if output_path:
        cv2.imwrite(output_path, canny)

    return canny


# CrossAttnStoreProcessor 클래스 정의 (SAG에 필요)
class CrossAttnStoreProcessor:
    def __init__(self):
        # 어텐션 확률을 저장할 변수 초기화
        self.attention_probs = None

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        # Query 생성
        query = attn.to_q(hidden_states)

        # Encoder Hidden States 설정
        encoder_hidden_states = encoder_hidden_states or hidden_states
        if attn.norm_cross and encoder_hidden_states is not hidden_states:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        # Key와 Value 생성
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Query, Key, Value를 헤드 차원으로 변환
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 어텐션 확률 계산 및 저장
        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 어텐션 가중치와 Value를 곱하여 Hidden States 업데이트
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 선형 변환 및 드롭아웃 적용
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# BLDCSAG1024 클래스 정의
class BLDCSAG512:
    def __init__(
        self,
        prompt,
        negative_prompt,
        blending_start_percentage,
        device,
        batch_size=1,
        output_path="output.png",
        init_image=None,
        mask=None,
    ):
        # 초기화: 입력된 파라미터들을 인스턴스 변수로 저장
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.init_image = init_image
        self.mask = mask
        self.model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.controlnet_model_path = "lllyasviel/control_v11p_sd15_canny"
        self.blending_start_percentage = blending_start_percentage
        self.device = device
        self.output_path = output_path
        self.latent_list = []
        self.degraded_latents_list = []
        self.batch_size = batch_size  # 배치 크기 설정

        # 모델 로드 메서드 호출
        self.load_models()

    def load_models(self):
        # Stable Diffusion 2.1 파이프라인 로드 (UNet 및 ControlNet 포함)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_path,
            controlnet=ControlNetModel.from_pretrained(
                self.controlnet_model_path,
                torch_dtype=torch.float16,
            ),
            torch_dtype=torch.float16,
        ).to(self.device)

        # VAE, UNet, 텍스트 인코더, 토크나이저, 스케줄러 로드
        self.vae = self.pipe.vae.to(self.device)
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_path, subfolder="scheduler"
        )

    def load_file(file_path):
        if file_path.endswith(".safetensors"):
            from safetensors.torch import load_file as _load_file

            return _load_file(file_path)
        else:
            return torch.load(file_path, map_location="cpu")

    @torch.no_grad()
    def edit_image(
        self,
        height=512,
        width=512,
        kernel_size=1,
        num_inference_steps=50,
        guidance_scale=7.0,
        generator=None,
        sag_scale=0.6,
        seed=0,
    ):
        # 랜덤 시드 설정
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # 배치 크기 설정
        batch_size = self.batch_size
        if isinstance(self.prompt, str):
            prompts = [self.prompt] * batch_size
        elif isinstance(self.prompt, list):
            assert len(self.prompt) == batch_size
            prompts = self.prompt
        else:
            raise ValueError("prompt must be a string or a list of strings")

        if isinstance(self.negative_prompt, str):
            negative_prompts = [self.negative_prompt] * batch_size
        elif isinstance(self.negative_prompt, list):
            assert len(self.negative_prompt) == batch_size
            negative_prompts = self.negative_prompt
        else:
            raise ValueError("negative_prompt must be a string or a list of strings")

        # 2. 이미지 로드 및 리사이즈
        image = (
            Image.open(self.init_image)
            .convert("RGB")
            .resize((width, height), Image.BILINEAR)
        )
        image_np = np.array(image)

        # 3. Canny Edge 이미지 생성
        canny_image = self._create_canny_image(image_np)
        Image.fromarray(canny_image).save("canny.png")
        controlnet_cond = self._prepare_control_image(canny_image)

        # 4. 원본 이미지를 latent space로 변환
        source_latents = self._image2latent(image)

        # 5. 마스크 로드 및 처리
        latent_mask = self._read_mask(self.mask, dest_size=(height // 8, width // 8))
        mask_np = latent_mask.squeeze().cpu().numpy().astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        eroded_mask = (
            torch.tensor(eroded_mask_np)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
            .half()
        )
        
        # # original mask에 ersion 적용
        # original_mask = self._read_mask(self.mask, dest_size=(height, width))
        # mask_np = original_mask.squeeze().cpu().numpy().astype(np.uint8)
        # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # eroded_mask_np = cv2.erode(mask_np, kernel, iterations=1)
        # latent_size = (height // 8, width // 8)
        # latent_mask_np = cv2.resize(eroded_mask_np, latent_size, interpolation=cv2.INTER_NEAREST)
        # eroded_mask = (
        #     torch.tensor(latent_mask_np)
        #     .unsqueeze(0)
        #     .unsqueeze(0)
        #     .to(self.device)
        #     .half()
        # )
        ############

        # 6. 텍스트 임베딩 생성
        text_embeddings = self._get_text_embeddings(prompts)
        uncond_embeddings = self._get_text_embeddings(negative_prompts)

        # 초기 Latent 설정
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=torch.float16,
        )

        # 타임스텝 설정
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 텐서 크기 맞추기
        source_latents = source_latents.repeat(batch_size, 1, 1, 1)
        controlnet_cond = (
            controlnet_cond.repeat(batch_size, 1, 1, 1).to(self.device).half()
        )

        # SAG를 위한 변수 초기화
        store_processor = CrossAttnStoreProcessor()
        original_attn_processors = self.unet.attn_processors
        map_size = None

        # 맵 사이즈를 얻기 위한 후크 함수
        def get_map_size(module, input, output):
            nonlocal map_size
            if isinstance(output, tuple):  # 출력이 tuple 형태인지 확인
                output_tensor = output[0]  # tuple의 첫 번째 tensor를 가져옴
            else:
                output_tensor = output  # tuple이 아니면 그대로 사용
            map_size = output_tensor.shape[
                -2:
            ]  # tensor의 마지막 두 차원 (H, W)를 가져옴

        # 어텐션 프로세서와 후크 등록
        self.unet.mid_block.attentions[0].transformer_blocks[
            0
        ].attn1.processor = store_processor
        self.unet.mid_block.attentions[0].register_forward_hook(get_map_size)

        # 타임스텝 루프 시작
        blending_start_step = int(len(timesteps) * self.blending_start_percentage)
        for i, t in enumerate(timesteps):
            # Latent 모델 입력 스케일링
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # CFG를 위해 복제
            latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
            controlnet_cond_in = torch.cat([controlnet_cond] * 2, dim=0)
            combined_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

            # ControlNet 적용
            controlnet_output = self.pipe.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=combined_embeddings,
                controlnet_cond=controlnet_cond_in,
                return_dict=False,
                conditioning_scale = 0.8,
            )

            # UNet 적용
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=combined_embeddings,
                down_block_additional_residuals=controlnet_output[0],
                mid_block_additional_residual=controlnet_output[1],
            ).sample

            # Conditional & Unconditional 분리 및 CFG 적용
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # 어텐션 맵 저장
            uncond_attn, cond_attn = store_processor.attention_probs.chunk(2, dim=0)
            uncond_attn = uncond_attn.detach()
            # 어텐션 확률 초기화
            store_processor.attention_probs = None

            # SAG 적용
            if sag_scale > 0.0:
                # x0와 epsilon 예측
                pred_x0 = self.pred_x0(latents, noise_pred_uncond, t)
                eps = self.pred_epsilon(latents, noise_pred_uncond, t)

                # SAG 마스킹
                degraded_latents, attn_mask = self.sag_masking(
                    pred_x0, uncond_attn, map_size, t, eps
                )

                # Degraded 입력 준비
                degraded_latent_model_input = self.scheduler.scale_model_input(
                    degraded_latents, t
                )

                # Degraded 입력에 대한 ControlNet 적용
                degraded_controlnet_output = self.pipe.controlnet(
                    degraded_latent_model_input,
                    t,
                    encoder_hidden_states=uncond_embeddings,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                    conditioning_scale = 0.8,
                )

                # UNet 적용 (Unconditional embeddings 사용)
                degraded_noise_pred = self.unet(
                    degraded_latent_model_input,
                    t,
                    encoder_hidden_states=uncond_embeddings,
                    down_block_additional_residuals=degraded_controlnet_output[0],
                    mid_block_additional_residual=degraded_controlnet_output[1],
                ).sample

                # noise_pred 업데이트
                noise_pred += sag_scale * (noise_pred - degraded_noise_pred)
                # noise_pred = degraded_noise_pred + sag_scale * (noise_pred - degraded_noise_pred)
            # latents 업데이트

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            noise_source_latents = self.scheduler.add_noise(
                source_latents, torch.randn_like(latents), t
            )
            # 마스크와 블렌딩
            if i >= blending_start_step:
                latents = latents * eroded_mask + noise_source_latents * (
                    1 - eroded_mask
                )

            # self.latent_list.append(attn_mask)

        # for i in range(num_inference_steps):
        #     # 새 figure를 만듦
        #     fig, ax = plt.subplots(figsize=(6, 6))
        #     fig.suptitle(f"{i+1} step")

        #     # 첫 번째 채널만 가져옴 (필요 시 채널 인덱스 변경 가능)
        #     channel_index = 0
        #     image_tensor = self.latent_list[i].squeeze().cpu().numpy()

        #     # 이미지 출력
        #     ax.imshow(image_tensor[channel_index], cmap='gray')
        #     ax.axis('off')

        #     # 여백 조정 및 저장
        #     plt.tight_layout()
        #     plt.subplots_adjust(top=0.85)  # 제목 공간 확보

        #     # 파일명 설정 및 저장
        #     filename = f'images/{i+1}.png'
        #     plt.savefig(filename)  # 이미지 파일 저장
        #     plt.close(fig)

        # 원래의 어텐션 프로세서로 복원
        self.unet.set_attn_processor(original_attn_processors)

        # Latents를 이미지로 디코딩
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        # 후처리 및 반환
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _image2latent(self, image):
        # 이미지를 텐서로 변환하고 정규화
        image = np.array(image).astype(np.float32) / 127.5 - 1
        image = (
            torch.from_numpy(image.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
            .half()
        )
        # VAE를 통해 Latent로 인코딩
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def _read_mask(self, mask_path: str, dest_size=(128, 128)):
        # 마스크 이미지 로드 및 이진화
        mask = Image.open(mask_path).convert("L").resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255.0
        mask = (mask >= 0.5).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).half()
        return mask

    def _create_canny_image(self, image):
        # OpenCV를 사용하여 Canny Edge 이미지 생성
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 50, 250)
        return edges

    def _prepare_control_image(self, image):
        # 이미지를 텐서로 변환 및 전처리
        if not isinstance(image, torch.Tensor):
            image = np.array(image)
            if len(image.shape) == 2:
                image = image[:, :, None]
            image = image.transpose(2, 0, 1)  # (C, H, W)
            if image.shape[0] == 1:
                image = np.repeat(image, repeats=3, axis=0)
            image = image / 255.0
            image = torch.from_numpy(image).float()
        return image.to(self.device).half()

    def _get_text_embeddings(self, texts):
        # 텍스트 임베딩 생성
        text_input = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    # pred_x0 함수 정의
    def pred_x0(self, sample, model_output, timestep):
        # 알파 및 베타 값 계산
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        # 예측 타입에 따라 pred_original_sample 계산
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t**0.5 * model_output
            ) / alpha_prod_t**0.5
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (
                alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
            )
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )
        return pred_original_sample

    # pred_epsilon 함수 정의
    def pred_epsilon(self, sample, model_output, timestep):
        # 알파 및 베타 값 계산
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        # 예측 타입에 따라 pred_eps 계산
        if self.scheduler.config.prediction_type == "epsilon":
            pred_eps = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_eps = (sample - alpha_prod_t**0.5 * model_output) / beta_prod_t**0.5
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_eps = beta_prod_t**0.5 * sample + alpha_prod_t**0.5 * model_output
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )
        return pred_eps

    # sag_masking 함수 정의
    def sag_masking(self, original_latents, attn_map, map_size, t, eps):
        # 어텐션 맵의 크기 및 Latent 크기 가져오기
        bh, hw1, hw2 = attn_map.shape
        b = original_latents.shape[0]
        latent_channel = original_latents.shape[1]
        latent_h = original_latents.shape[2]
        latent_w = original_latents.shape[3]
        h = self.unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]

        # 어텐션 마스크 생성
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1).sum(1) > 1.0
        attn_mask = (
            attn_mask.reshape(b, map_size[0], map_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w), mode="nearest")

        # 블러 적용
        degraded_latents = self.gaussian_blur_2d(
            original_latents, kernel_size=9, sigma=1.0
        )
        degraded_latents = degraded_latents * attn_mask + original_latents * (
            1 - attn_mask
        )
        # degraded_latents = degraded_latents * (1 - attn_mask) + original_latents * attn_mask

        # 노이즈 추가
        degraded_latents = self.scheduler.add_noise(
            degraded_latents, noise=eps, timesteps=t[None]
        )

        return degraded_latents, attn_mask

    # gaussian_blur_2d 함수 정의
    def gaussian_blur_2d(self, img, kernel_size, sigma):
        # 가우시안 커널 생성
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        x_kernel = pdf / pdf.sum()
        x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)
        kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
        kernel2d = kernel2d.expand(
            img.shape[1], 1, kernel2d.shape[0], kernel2d.shape[1]
        )
        padding = kernel_size // 2
        # 이미지에 패딩 적용 및 컨볼루션
        img = F.pad(img, (padding, padding, padding, padding), mode="reflect")
        img = F.conv2d(img, kernel2d, groups=img.shape[1])
        return img


# SamImageProcessor 클래스 정의
class SamImageProcessor:
    def __init__(self, img_path, mask_num, device, model_name="facebook/sam-vit-huge"):
        self.img_path = img_path
        self.device = device
        # 모델 및 프로세서 로드
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
        # 이미지 로드
        self.pf_image = self._load_image()
        self.mask_num = mask_num % 3  # 마스크 번호 설정

    def _load_image(self):
        # 이미지 로드 및 리사이즈
        pf_image = (
            Image.open(self.img_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        )
        return pf_image

    def process_image(self):
        # 입력 포인트 설정 (이미지 중앙)
        input_points = [[[self.pf_image.size[0] // 2, self.pf_image.size[1] // 2]]]
        # 입력 데이터 생성
        inputs = self.processor(
            self.pf_image, input_points=input_points, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            # 모델 추론
            outputs = self.model(**inputs)
        return inputs, outputs

    def post_process_masks(self, inputs, outputs):
        # 마스크 후처리
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return masks

    def extract_max_score_mask(self, masks):
        # 최대 점수의 마스크 추출
        max_score_mask = masks[0][:, self.mask_num, :, :]
        return max_score_mask

    def save_mask_image(self, max_score_mask):
        # 마스크 이미지를 저장
        mask_image = np.where(
            max_score_mask.numpy().reshape(1024, 1024, 1) != 0,
            [0, 0, 0, 255],
            [255, 255, 255, 0],
        ).astype(np.uint8)
        mask_image_pil = Image.fromarray(mask_image, "RGBA")
        mask_image_pil.save("mask.png")

    def save_masked_image(self, max_score_mask):
        # 마스크된 이미지를 저장
        mask_resized = max_score_mask.squeeze().numpy().astype(np.uint8)
        masked_image = np.array(self.pf_image) * mask_resized[:, :, np.newaxis]

        alpha_channel = (mask_resized * 255).astype(np.uint8)
        masked_image_rgba = np.dstack((masked_image, alpha_channel))

        masked_image_pil = Image.fromarray(masked_image_rgba, "RGBA")
        masked_image_pil.save("image.png")

    def run(self):
        # 전체 프로세스 실행
        inputs, outputs = self.process_image()
        masks = self.post_process_masks(inputs, outputs)
        max_score_mask = self.extract_max_score_mask(masks)
        self.save_mask_image(max_score_mask)
        self.save_masked_image(max_score_mask)


from segment_anything_hq import sam_model_registry, SamPredictor


class SamHQImageProcessor:
    def __init__(
        self,
        img_path,
        device,
        coord=(2, 2, 14, 14),
        model_type="vit_h",
        sam_checkpoint="/home/ada6k4_01/Desktop/inpainting/pretrained_checkpoint/sam_hq_vit_h.pth",
    ):
        self.img_path = img_path
        self.device = device
        # SAM 모델과 predictor 로드
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.load_state_dict(torch.load(sam_checkpoint), strict=False)
        self.x1, self.y1, self.x2, self.y2 = coord
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        # 이미지 로드
        self.pf_image = self._load_image()

    def _load_image(self):
        # 이미지 로드 및 준비
        image = cv2.imread(self.img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def process_image(self):
        # predictor에 이미지 설정
        self.predictor.set_image(self.pf_image)

        # 경계 박스를 입력 프롬프트로 사용하여 세그멘테이션 수행
        h, w, _ = self.pf_image.shape
        box = np.array(
            [[self.x1 * w / 16, self.y1 * h / 16], [self.x2 * w / 16, self.y2 * h / 16]]
        )

        # 마스크 예측
        masks, scores, logits = self.predictor.predict(
            box=box, multimask_output=True  # 경계 박스를 프롬프트로 사용
        )
        # 점수가 가장 높은 마스크 선택
        max_score_index = np.argmax(scores)
        max_score_mask = masks[max_score_index]
        return max_score_mask

    def save_mask_image(self, max_score_mask):
        # 마스크 이미지를 저장
        mask_resized = max_score_mask.astype(np.uint8)
        mask_image = np.where(
            mask_resized.reshape(mask_resized.shape[0], mask_resized.shape[1], 1) != 0,
            [0, 0, 0, 255],
            [255, 255, 255, 0],
        ).astype(np.uint8)
        mask_image_pil = Image.fromarray(mask_image, "RGBA")
        mask_image_pil.save("mask.png")

    def save_masked_image(self, max_score_mask):
        # 마스크된 이미지를 저장
        mask_resized = max_score_mask.astype(np.uint8)
        masked_image = self.pf_image * mask_resized[:, :, np.newaxis]
        alpha_channel = (mask_resized * 255).astype(np.uint8)
        masked_image_rgba = np.dstack((masked_image, alpha_channel))
        masked_image_pil = Image.fromarray(masked_image_rgba, "RGBA")
        masked_image_pil.save("image.png")

    def run(self):
        # 전체 프로세스를 실행
        max_score_mask = self.process_image()
        self.save_mask_image(max_score_mask)
        self.save_masked_image(max_score_mask)


class ImageGrid:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.width, self.height = self.img.size
        self.grid_size_x = self.width / 16
        self.grid_size_y = self.height / 16

    def draw_grid(self):
        # 이미지와 격자 표시하기
        fig, ax = plt.subplots()
        ax.imshow(self.img)

        # 격자 그리기
        for i in range(17):  # 16개의 셀을 위해 17개의 선이 필요
            # 수직선 그리기
            ax.add_line(
                plt.Line2D(
                    (i * self.grid_size_x, i * self.grid_size_x),
                    (0, self.height),
                    color="red",
                )
            )
            # 수평선 그리기
            ax.add_line(
                plt.Line2D(
                    (0, self.width),
                    (i * self.grid_size_y, i * self.grid_size_y),
                    color="red",
                )
            )

        # 격자 번호 추가
        for i in range(17):
            # 왼쪽에 행 번호 추가 (0부터 시작)
            ax.text(
                -self.grid_size_x / 2,
                (i) * self.grid_size_y,
                str(i),
                va="center",
                ha="center",
                color="blue",
                fontsize=8,
            )
            # 위쪽에 열 번호 추가 (0부터 시작)
            ax.text(
                (i) * self.grid_size_x,
                -self.grid_size_y / 2,
                str(i),
                va="center",
                ha="center",
                color="blue",
                fontsize=8,
            )

        # 표시하기
        plt.axis("off")
        plt.show()


# ImageGridDisplay 클래스 정의
class ImageGridDisplay:
    def __init__(self, img1_path, img2_path, img3_path, img4):
        # 이미지 경로 저장
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.img3_path = img3_path
        self.img4 = img4
        self.images = []
        # 이미지 로드 메서드 호출
        self.load_images()

    def load_images(self):
        # 이미지를 OpenCV로 로드하고 RGB로 변환하여 저장
        self.images.append(cv2.cvtColor(cv2.imread(self.img1_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img2_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img3_path), cv2.COLOR_BGR2RGB))
        self.images.append(Image.fromarray(self.img4))

    def display(self):
        # 이미지를 2x2 그리드로 디스플레이
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        titles = ["Original", "Mask", "Canny", "Output"]
        for i, ax in enumerate(axs.flat):
            ax.imshow(self.images[i])
            ax.set_title(titles[i])
            ax.axis("off")

        plt.tight_layout()
        plt.show()


class ImageGridSaver:
    def __init__(self, images, s, t, save_path):
        self.images = images
        self.s = s
        self.t = t
        self.save_path = save_path
        self.n, self.h, self.w, self.c = images.shape

        # 이미지 개수와 s*t 일치 여부 검사
        if self.n != self.s * self.t:
            raise ValueError("s * t가 이미지 개수와 일치하지 않습니다.")

    def create_grid(self):
        # 각 행(row)별 이미지를 연결하여 results에 추가
        results = []
        for row in range(self.t):
            row_images = self.images[row * self.s : (row + 1) * self.s]
            row_concat = np.concatenate(row_images, axis=1)  # 가로로 연결
            results.append(row_concat)

        # 모든 행을 세로로 연결하여 최종 그리드 생성
        results_flat = np.concatenate(results, axis=0)
        return results_flat

    def save_grid(self):
        # 이미지 그리드 생성 후 저장
        grid_image = self.create_grid()
        result_image = Image.fromarray(grid_image)  # 이미지 범위(0-255)로 변환
        result_image.save(self.save_path)
        print(f"이미지가 {self.save_path}에 저장되었습니다.")


class ParametersTable:
    def __init__(self, parameters):
        self.parameters = parameters
        self.processed_data = self.wrap_text()  # Prepare wrapped text for display

    def wrap_text(self):
        # Wrap text for longer fields
        wrapped_prompt = "\n".join(textwrap.wrap(self.parameters["prompt"], width=50))
        wrapped_negative_prompt = "\n".join(
            textwrap.wrap(self.parameters["negative_prompt"], width=50)
        )

        # Determine max height for the table
        max_height = max(
            wrapped_prompt.count("\n"), wrapped_negative_prompt.count("\n")
        )

        # Seed handling
        seed = (
            "no guidance"
            if self.parameters["sag_scale"] <= 0.0
            else self.parameters["sag_scale"]
        )

        # Organize table data
        table_data = [
            ["Prompt", wrapped_prompt],
            ["Negative Prompt", wrapped_negative_prompt],
            ["Blending Start Percentage", self.parameters["blending_start_percentage"]],
            ["Kernel Size", self.parameters["kernel_size"]],
            ["Inference Steps", self.parameters["num_inference_steps"]],
            ["Guidance Scale", self.parameters["guidance_scale"]],
            ["SAG Scale", self.parameters["sag_scale"]],
            ["Seed", self.parameters["seed"]],
            ["Generation time", round(self.parameters["generation_time"], 2)],
        ]

        return table_data, max_height

    def display_table(self):
        table_data, max_height = self.processed_data

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        # Define column widths
        col_widths = [0.35, 0.65]

        # Create the table with specified column widths
        table = ax.table(
            cellText=table_data,
            colLabels=None,
            cellLoc="left",
            loc="center",
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, max_height + 2)

        # Set background color for the left column to light gray
        for i in range(len(table_data)):
            cell = table[(i, 0)]
            cell.set_facecolor("#f0f0f0")

        plt.show()


class ImageQualityEvaluator:
    def __init__(self, output_path, mask_path, original_path):
        # Store image paths
        self.output_path = output_path
        self.mask_path = mask_path
        self.original_path = original_path

        # Load images
        self.original_image = (
            Image.open(original_path)
            .convert("RGBA")
            .resize((1024, 1024), Image.BILINEAR)
        )
        self.output_image = (
            Image.open(output_path).convert("RGBA").resize((1024, 1024), Image.BILINEAR)
        )
        self.mask_image = (
            Image.open(mask_path).convert("RGBA").resize((1024, 1024), Image.BILINEAR)
        )

        # Apply mask to the output image
        self.generated_image = self.apply_mask(self.output_image, self.mask_image)

        # Convert images to RGB for metric computations
        self.original_rgb = self.original_image.convert("RGB")
        self.generated_rgb = self.generated_image.convert("RGB")

        # Pre-load models
        self.lpips_fn = lpips.LPIPS(net="vgg")  # LPIPS model
        self.iqa_model = pyiqa.create_metric("musiq")  # IQA model

    def apply_mask(self, output_image, mask_image):
        # Convert images to numpy arrays
        output_array = np.array(output_image).astype(np.float32)
        mask_array = np.array(mask_image).astype(np.float32)

        # Normalize the alpha channel of the mask
        alpha_mask = mask_array[..., 3:4] / 255.0  # Shape: (H, W, 1)

        # Apply the mask to the RGB channels
        masked_array = output_array.copy()
        masked_array[..., :3] *= alpha_mask

        # Clip values to [0, 255] and convert to uint8
        masked_array = np.clip(masked_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(masked_array, "RGBA")

    def image_to_tensor(self, image):
        # Convert PIL image to torch tensor
        image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        tensor = (
            torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        )  # Shape: (1, C, H, W)
        return tensor

    def compute_psnr(self):
        # Compute PSNR between original and generated images
        psnr_value = psnr(np.array(self.original_rgb), np.array(self.generated_rgb))
        return psnr_value

    def compute_ssim(self):
        # Compute SSIM between original and generated images
        ssim_value = ssim(
            np.array(self.original_rgb), np.array(self.generated_rgb), channel_axis=2
        )
        return ssim_value

    def compute_lpips(self):
        # Compute LPIPS between original and generated images
        original_tensor = self.image_to_tensor(self.original_rgb)
        generated_tensor = self.image_to_tensor(self.generated_rgb)
        with torch.no_grad():
            lpips_value = self.lpips_fn(original_tensor, generated_tensor)
        return lpips_value.item()

    def compute_brisque(self):
        # Compute BRISQUE score for the output image
        output_cv = cv2.imread(self.output_path, cv2.IMREAD_COLOR)
        if output_cv is None:
            raise FileNotFoundError(f"Could not read image at {self.output_path}")
        score = brisque.BRISQUE().score(output_cv)
        return score

    def compute_iqa(self):
        # Compute IQA score for the generated image using MusIQ
        with torch.no_grad():
            score = self.iqa_model(self.generated_rgb)
        return score.item()

    def compute_rgb_histogram_similarity(self):
        # Convert PIL images to numpy arrays in RGB format
        original_rgb = np.array(self.original_image.convert("RGB"))
        generated_rgb = np.array(self.generated_image.convert("RGB"))

        # Calculate histograms for each channel and normalize
        histograms_original = []
        histograms_generated = []
        for channel in range(3):  # RGB channels
            hist_orig = cv2.calcHist([original_rgb], [channel], None, [256], [0, 256])
            hist_gen = cv2.calcHist([generated_rgb], [channel], None, [256], [0, 256])

            # Normalize histograms
            hist_orig = cv2.normalize(hist_orig, hist_orig).flatten()
            hist_gen = cv2.normalize(hist_gen, hist_gen).flatten()

            histograms_original.append(hist_orig)
            histograms_generated.append(hist_gen)

        # Calculate cosine similarity for each channel
        def cosine_similarity(hist1, hist2):
            return np.dot(hist1, hist2) / (
                np.linalg.norm(hist1) * np.linalg.norm(hist2)
            )

        # Calculate similarities with weights emphasizing low similarity scores
        similarities = [
            cosine_similarity(h_orig, h_gen)
            for h_orig, h_gen in zip(histograms_original, histograms_generated)
        ]

        # Apply weights: assign greater weight to channels with larger differences
        # Use (1 - similarity) as weight to emphasize differences
        weights = [1 - similarity for similarity in similarities]
        weighted_average_similarity = sum(
            sim * weight for sim, weight in zip(similarities, weights)
        ) / sum(weights)

        return weighted_average_similarity

    def compute_all_metrics(self):
        # Compute all metrics and return as a dictionary
        metrics = {
            "PSNR": self.compute_psnr(),
            "SSIM": self.compute_ssim(),
            "LPIPS": self.compute_lpips(),
            "BRISQUE": self.compute_brisque(),
            "IQA": self.compute_iqa(),
        }
        return metrics

    def show_metrics_table(self):
        # Get metric results
        metrics = self.compute_all_metrics()

        # Prepare data
        metrics_names = list(metrics.keys())
        metrics_values = [metrics[name] for name in metrics_names]

        # Create table
        fig, ax = plt.subplots(figsize=(6, 2))
        table_data = [
            [name, f"{value:.4f}"] for name, value in zip(metrics_names, metrics_values)
        ]
        table = ax.table(
            cellText=table_data, colLabels=["Metric", "Value"], loc="center"
        )

        # Set table style
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        cell = table[(0, 0)]
        cell.set_facecolor("#f0f0f0")
        cell = table[(0, 1)]
        cell.set_facecolor("#f0f0f0")
        ax.axis("off")

        # Display graph
        plt.show()
