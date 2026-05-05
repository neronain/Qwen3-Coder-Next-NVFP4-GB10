# 🚀 Qwen3-Coder-Next (NVFP4) on DGX Spark via vLLM

คู่มือการติดตั้งและรันโมเดล `saricles/Qwen3-Coder-Next-NVFP4-GB10` บนเครื่อง NVIDIA DGX Spark (1 Node, 128GB Unified Memory) ผ่าน vLLM Container สำหรับใช้งานร่วมกับระบบ Multi-Agent (เช่น OpenClaw / Claw-Empire)

## 📋 Prerequisites
*   **Hardware:** NVIDIA DGX Spark (รองรับสถาปัตยกรรม Blackwell GB10 และ NVFP4)
*   **OS:** Ubuntu 22.04 / 24.04 (มี Docker และ NVIDIA Container Toolkit ติดตั้งเรียบร้อยแล้ว)
*   **Hugging Face Account:** ต้องมีบัญชีและสร้าง Access Token (Read) เตรียมไว้

---

## 🛠️ Step 1: เตรียมความพร้อมและดาวน์โหลด Model

เนื่องจากโมเดลนี้เป็น Gated Model (ต้องกดยอมรับเงื่อนไขก่อนโหลด) ให้ทำตามขั้นตอนดังนี้:

### 1.1 ยอมรับ License บนหน้าเว็บ
1. ไปที่ลิงก์: [saricles/Qwen3-Coder-Next-NVFP4-GB10](https://huggingface.co/saricles/Qwen3-Coder-Next-NVFP4-GB10)
2. ล็อกอินและกดปุ่ม **"Agree and access repository"**
3. ไปที่ [Settings > Tokens](https://huggingface.co/settings/tokens) แล้วสร้าง Token ใหม่แบบ **Read** คัดลอกเก็บไว้

### 1.2 ติดตั้ง Hugging Face CLI (`hf`) บนเครื่อง DGX
หากเครื่องยังไม่มีคำสั่ง `hf` ให้รันคำสั่งต่อไปนี้เพื่อติดตั้งและเพิ่มลงใน PATH:
```bash
# ติดตั้ง huggingface-cli ข้ามข้อจำกัด externally-managed-environment ของ Ubuntu
pip3 install -U "huggingface_hub[cli]" --break-system-packages

# นำคำสั่งเข้าสู่ระบบ PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 1.3 ล็อกอินและดาวน์โหลดไฟล์โมเดล (~45GB)
```bash
# ล็อกอินเข้าสู่ Hugging Face (ระบบจะให้วาง Token ที่ก๊อปปี้มา)
hf auth login

# ดาวน์โหลดโมเดลมาเก็บไว้ในโฟลเดอร์ Home ของคุณ
hf download saricles/Qwen3-Coder-Next-NVFP4-GB10 --local-dir /home/neronain/models/Qwen3-Coder-Next-NVFP4-GB10
```

---

## 🐳 Step 2: สร้างไฟล์ Docker Compose

สร้างไฟล์ `docker-compose-qwen-code.yml` ในไดเรกทอรีที่คุณต้องการ โดยใช้ Configuration ที่ปรับจูนมาเพื่อรีดประสิทธิภาพ Blackwell / NVFP4 สูงสุด:

```yaml
services:
  coder-next:
    image: avarok/dgx-vllm-nvfp4-kernel:v23
    container_name: coder-next
    restart: unless-stopped
    ipc: host
    shm_size: 32gb
    ports:
      - "8000:8000"
    volumes:
      # Mount โมเดลที่ดาวน์โหลดไว้เข้าสู่ Container
      - /home/neronain/models/Qwen3-Coder-Next-NVFP4-GB10:/models/Qwen3-Coder-Next-NVFP4-GB10
    environment:
      # --- vLLM Blackwell/NVFP4 Backend Config ---
      - VLLM_NVFP4_GEMM_BACKEND=marlin
      - VLLM_TEST_FORCE_FP8_MARLIN=1
      - VLLM_USE_FLASHINFER_MOE_FP4=0
      - VLLM_MARLIN_USE_ATOMIC_ADD=1

      # --- Model & Server Config ---
      - MODEL=/models/Qwen3-Coder-Next-NVFP4-GB10
      - PORT=8000
      - MAX_MODEL_LEN=262144
      - GPU_MEMORY_UTIL=0.90

      # --- Extra VLLM Arguments ---
      # การใส่ --served-model-name qwen3_coder ทำให้ OpenClaw เรียกใช้งานได้ทันทีโดยไม่ต้องแก้ Path ยืดยาว
      - VLLM_EXTRA_ARGS=--served-model-name qwen3_coder --kv-cache-dtype fp8 --attention-backend flashinfer --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 --enable-auto-tool-choice --tool-call-parser qwen3_coder

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**🔥 ไฮไลท์การปรับแต่ง (Optimizations):**
*   `VLLM_NVFP4_GEMM_BACKEND=marlin`: บังคับใช้ Marlin backend ทำความเร็วคูณเมทริกซ์สูงสุดบน NVFP4
*   `--kv-cache-dtype fp8`: บีบอัด KV Cache ลงครึ่งหนึ่ง ทำให้รองรับ Context Length ได้ถึง 262K บนแรม 128GB
*   `--enable-chunked-prefill`: ป้องกันปัญหา Out of Memory (OOM) เมื่อส่ง Source code ยาวๆ ให้ Agent อ่าน
*   `--served-model-name qwen3_coder`: สร้าง Alias ให้โมเดลเพื่อให้ระบบปลายทางเชื่อมต่อง่ายขึ้น

---

## 🚀 Step 3: รันเซิร์ฟเวอร์ (Start vLLM Server)

สั่งรัน Container ด้วยคำสั่ง:
```bash
sudo docker compose -f docker-compose-qwen-code.yml up -d
```

เช็คสถานะการโหลดโมเดลเข้า VRAM (ใช้เวลาประมาณ 10-30 วินาทีเนื่องจากโหลดจาก Local SSD):
```bash
sudo docker logs -f coder-next
```
เมื่อปรากฏข้อความ `Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000)` แสดงว่าเซิร์ฟเวอร์พร้อมใช้งาน!

---

## 🔌 Step 4: การเชื่อมต่อกับ OpenClaw / Multi-Agent

นำ Endpoint นี้ไปตั้งค่าในระบบ Multi-Agent (เช่น Dev Agent) ของคุณ:
*   **Base URL (API):** `http://<DGX_IP>:8000/v1`
*   **Model Name:** `qwen3_coder`

---

หวังว่าสรุปนี้จะช่วยให้โปรเจกต์บน GitHub ของคุณดูโปรและเป็นประโยชน์กับนักพัฒนาระบบคนอื่นๆ นะครับ! ถ้ามีปรับจูนส่วนไหนเพิ่มเติมในอนาคต ทักมาปรึกษาได้เสมอครับ</DGX_IP>
