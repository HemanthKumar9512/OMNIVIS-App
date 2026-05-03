# OMNIVIS — Global Deployment Guide (FREE)

This guide walks you through deploying OMNIVIS globally for free using Google Colab (GPU backend) + Vercel (frontend CDN).

---

## Architecture

```
User → Vercel CDN (Frontend) → ngrok → Google Colab GPU (Backend)
         [Global, HTTPS]            [Tunnel]     [Free GPU, ML Inference]
```

---

## Part 1: Deploy Backend on Google Colab (Free GPU)

### Step 1: Get ngrok Auth Token

1. Sign up at https://dashboard.ngrok.com/signup (FREE)
2. Go to https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your auth token

### Step 2: Open Colab Notebook

1. Go to https://colab.research.google.com/
2. Upload the `deploy/colab-deployment.ipynb` file
3. OR open directly if pushed to GitHub

### Step 3: Run the Notebook

1. **Set your ngrok token** in the first code cell
2. **Upload backend files** — either clone from GitHub or upload the `backend` folder manually
3. **Install dependencies** — run the pip install cell
4. **Start server** — run the final cell

### Step 4: Copy the ngrok URL

After running the notebook, you'll see output like:
```
YOUR OMNIVIS BACKEND URL: https://a1b2c3d4.ngrok-free.app
```

**Copy this URL** — you'll need it for the frontend.

> **Important:** Keep the Colab tab open. If you close it, the server stops.

---

## Part 2: Deploy Frontend on Vercel (Free Global CDN)

### Option A: Deploy via Vercel Dashboard (Easiest)

1. Push your code to GitHub
2. Go to https://vercel.com/new
3. Import your GitHub repository
4. Set the environment variable:
   - **Name:** `VITE_BACKEND_URL`
   - **Value:** Your ngrok URL (e.g., `https://a1b2c3d4.ngrok-free.app`)
5. Click **Deploy**
6. Your app will be live at `https://your-app.vercel.app`

### Option B: Deploy via CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to frontend directory
cd frontend

# Deploy with environment variable
vercel --build-env VITE_BACKEND_URL=https://a1b2c3d4.ngrok-free.app

# For production deployment
vercel --prod --build-env VITE_BACKEND_URL=https://a1b2c3d4.ngrok-free.app
```

---

## Part 3: Using Your App

1. Open your Vercel URL (e.g., `https://your-app.vercel.app`)
2. Allow camera access when prompted
3. The frontend will connect to your Colab backend via ngrok
4. Start using OMNIVIS!

---

## Troubleshooting

### "WebSocket connection failed"

- Verify your ngrok URL is correct in `VITE_BACKEND_URL`
- Make sure the Colab notebook is still running
- Check that ngrok hasn't expired (free URLs change on restart)

### "GPU not detected"

- In Colab, ensure you have a GPU assigned: Runtime → Change runtime type → GPU
- Free tier doesn't guarantee GPU availability

### Session disconnected

- Colab free tier disconnects after ~12 hours of inactivity
- Re-run all cells to restart
- The ngrok URL will change — update `VITE_BACKEND_URL` in Vercel and redeploy

### Camera access denied

- Camera requires HTTPS (Vercel provides this automatically)
- Check browser permissions for your site

---

## Limitations of Free Tier

| Service | Limitation |
|---------|------------|
| Google Colab | ~12hr session, must keep tab open, GPU not guaranteed |
| ngrok | URL changes on restart, rate limits, free plan limits |
| Vercel | 100GB bandwidth/month (plenty for most use cases) |

---

## Upgrading Later (If Needed)

When you're ready for more permanent hosting:

| Option | Cost | GPU | 24/7 |
|--------|------|-----|------|
| Oracle Cloud Free | $0 | No | Yes |
| RunPod | ~$0.20/hr | Yes | Yes |
| Lambda Labs | ~$0.50/hr | Yes | Yes |
| AWS g4dn.xlarge | ~$0.53/hr | Yes | Yes |

---

## Quick Reference Commands

### Re-deploy frontend after ngrok URL change
```bash
cd frontend
vercel --prod --build-env VITE_BACKEND_URL=https://new-ngrok-url.ngrok-free.app
```

### Check Colab GPU status
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Prevent Colab from auto-disconnecting
Paste in browser console:
```javascript
function ClickConnect() {
  document.querySelector('colab-toolbar-button')?.click();
}
setInterval(ClickConnect, 60000);
```
