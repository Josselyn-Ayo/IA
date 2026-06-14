// ============================
// CryptoPunks GAN - JavaScript Implementation
// Same model architecture and weights as generate_punks.py
// ============================

let modelWeights = null;
let modelLoaded = false;
let currentSeed = null;

// --- Tensor operations for the GAN forward pass ---

function convTranspose2d(input, N, C_in, H_in, W_in, weight, C_out, kH, kW, stride, padding) {
    const H_out = (H_in - 1) * stride - 2 * padding + kH;
    const W_out = (W_in - 1) * stride - 2 * padding + kW;
    const output = new Float32Array(N * C_out * H_out * W_out);

    for (let n = 0; n < N; n++) {
        for (let ic = 0; ic < C_in; ic++) {
            for (let ih = 0; ih < H_in; ih++) {
                for (let iw = 0; iw < W_in; iw++) {
                    const inIdx = n * C_in * H_in * W_in + ic * H_in * W_in + ih * W_in + iw;
                    const val = input[inIdx];
                    for (let oc = 0; oc < C_out; oc++) {
                        for (let kh = 0; kh < kH; kh++) {
                            for (let kw = 0; kw < kW; kw++) {
                                const oh = ih * stride - padding + kh;
                                const ow = iw * stride - padding + kw;
                                if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                                    const wIdx = ic * C_out * kH * kW + oc * kH * kW + kh * kW + kw;
                                    const oIdx = n * C_out * H_out * W_out + oc * H_out * W_out + oh * W_out + ow;
                                    output[oIdx] += val * weight[wIdx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return { data: output, shape: [N, C_out, H_out, W_out] };
}

function batchNorm2d(input, N, C, H, W, bnWeight, bnBias, runningMean, runningVar) {
    const output = new Float32Array(N * C * H * W);
    const eps = 1e-5;

    for (let c = 0; c < C; c++) {
        const invStd = bnWeight[c] / Math.sqrt(runningVar[c] + eps);
        const negMeanScaled = -runningMean[c] * invStd;
        const shift = bnBias[c];
        const coeff = invStd;
        const bias_val = negMeanScaled * coeff + shift;

        for (let n = 0; n < N; n++) {
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    const idx = n * C * H * W + c * H * W + h * W + w;
                    output[idx] = input[idx] * coeff + bias_val;
                }
            }
        }
    }

    return { data: output, shape: [N, C, H, W] };
}

function relu(input) {
    const output = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
    return output;
}

function tanh(input) {
    const output = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
        output[i] = Math.tanh(input[i]);
    }
    return output;
}

// --- GAN Forward Pass ---

function ganForward(noise, weights) {
    let x = { data: noise, shape: [1, 100, 1, 1] };

    // Layer 0: ConvTranspose2d(100, 256, 3, 1, 0)
    x = convTranspose2d(x.data, 1, 100, 1, 1,
        weights['network.0.weight'], 256, 3, 3, 1, 0);

    // Layer 1: BatchNorm2d(256)
    x = batchNorm2d(x.data, 1, 256, 3, 3,
        weights['network.1.weight'], weights['network.1.bias'],
        weights['network.1.running_mean'], weights['network.1.running_var']);

    // Layer 2: ReLU
    x.data = relu(x.data);
    x.shape = [1, 256, 3, 3];

    // Layer 3: ConvTranspose2d(256, 128, 3, 2, 1)
    x = convTranspose2d(x.data, 1, 256, 3, 3,
        weights['network.3.weight'], 128, 3, 3, 2, 1);

    // Layer 4: BatchNorm2d(128)
    x = batchNorm2d(x.data, 1, 128, 5, 5,
        weights['network.4.weight'], weights['network.4.bias'],
        weights['network.4.running_mean'], weights['network.4.running_var']);

    // Layer 5: ReLU
    x.data = relu(x.data);
    x.shape = [1, 128, 5, 5];

    // Layer 6: ConvTranspose2d(128, 64, 4, 2, 0)
    x = convTranspose2d(x.data, 1, 128, 5, 5,
        weights['network.6.weight'], 64, 4, 4, 2, 0);

    // Layer 7: BatchNorm2d(64)
    x = batchNorm2d(x.data, 1, 64, 12, 12,
        weights['network.7.weight'], weights['network.7.bias'],
        weights['network.7.running_mean'], weights['network.7.running_var']);

    // Layer 8: ReLU
    x.data = relu(x.data);
    x.shape = [1, 64, 12, 12];

    // Layer 9: ConvTranspose2d(64, 4, 4, 2, 1)
    x = convTranspose2d(x.data, 1, 64, 12, 12,
        weights['network.9.weight'], 4, 4, 4, 2, 1);

    // Layer 10: Tanh
    x.data = tanh(x.data);

    return x;
}

// --- Random noise generation (Box-Muller for normal distribution) ---

function randn(seed) {
    const rng = mulberry32(seed);
    const result = [];
    for (let i = 0; i < 100; i++) {
        let u1, u2;
        do { u1 = rng(); } while (u1 === 0);
        u2 = rng();
        result.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
    }
    return result;
}

function mulberry32(a) {
    return function() {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// --- Weight loading from JSON ---

function decodeWeight(w) {
    const binaryString = atob(w.data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    const buffer = bytes.buffer;
    if (w.dtype === 'float64') {
        return new Float32Array(new Float64Array(buffer));
    }
    return new Float32Array(buffer);
}

async function loadModel() {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const btnGenerate = document.getElementById('btnGenerate');
    const genMode = document.getElementById('genMode');
    const placeholder = document.getElementById('placeholder');
    const placeholderMsg = document.getElementById('placeholderMsg');

    try {
        statusText.textContent = 'Cargando modelo GAN (pesos)...';

        const response = await fetch('model_weights.json');
        if (!response.ok) throw new Error('model_weights.json no encontrado');

        statusText.textContent = 'Decodificando pesos...';
        const raw = await response.json();

        statusText.textContent = 'Preparando modelo...';
        modelWeights = {};
        for (const [name, w] of Object.entries(raw)) {
            modelWeights[name] = decodeWeight(w);
        }

        // Quick sanity check
        const testNoise = new Float32Array(100);
        const testOut = ganForward(testNoise, modelWeights);
        if (testOut.shape[0] !== 1 || testOut.shape[1] !== 4 || testOut.shape[2] !== 24 || testOut.shape[3] !== 24) {
            throw new Error('Model output shape mismatch');
        }

        modelLoaded = true;
        statusDot.className = 'status-indicator ready';
        statusText.textContent = 'Modelo GAN cargado - Listo para generar';
        btnGenerate.disabled = false;
        btnGenerate.innerHTML = '<span class="btn-icon">&#9889;</span> Generar Punks';
        genMode.textContent = 'Modo: GAN Real';
        placeholderMsg.textContent = 'Presiona "Generar Punks" para crear tu coleccion';

        generateGallery();
        generateHero();

    } catch (err) {
        console.warn('Error cargando modelo:', err);
        modelLoaded = false;
        statusDot.className = 'status-indicator error';
        statusText.textContent = 'Sin modelo - Ejecuta python generate_punks.py primero';
        btnGenerate.disabled = false;
        btnGenerate.innerHTML = '<span class="btn-icon">&#9889;</span> Generar Punks';
        genMode.textContent = 'Modo: Sin modelo';
        placeholderMsg.innerHTML = 'Ejecuta <code>python generate_punks.py</code> para crear model_weights.json, luego sube todo a Vercel';
        placeholderMsg.classList.add('error-msg');
    }
}

// --- Rendering ---

function renderPunksToCanvas(canvas, punks, cols, scale) {
    const rows = Math.ceil(punks.length / cols);
    const pw = 24, ph = 24;
    const gap = 2;
    canvas.width = cols * (pw * scale + gap) - gap;
    canvas.height = rows * (ph * scale + gap) - gap;
    canvas.style.imageRendering = 'pixelated';
    canvas.style.maxWidth = '100%';
    canvas.style.height = 'auto';

    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < punks.length; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const ox = col * (pw * scale + gap);
        const oy = row * (ph * scale + gap);

        const punk = punks[i];
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = pw;
        tmpCanvas.height = ph;
        const tmpCtx = tmpCanvas.getContext('2d');
        tmpCtx.imageSmoothingEnabled = false;

        const imgData = tmpCtx.createImageData(pw, ph);
        for (let y = 0; y < ph; y++) {
            for (let x = 0; x < pw; x++) {
                const idx = (y * pw + x) * 4;
                const r_idx = 0 * ph * pw + y * pw + x;
                const g_idx = 1 * ph * pw + y * pw + x;
                const b_idx = 2 * ph * pw + y * pw + x;
                const a_idx = 3 * ph * pw + y * pw + x;

                let r = punk[r_idx];
                let g = punk[g_idx];
                let b = punk[b_idx];
                let a = punk[a_idx];

                // Tanh output is [-1, 1], normalize to [0, 255]
                r = Math.max(0, Math.min(255, Math.round((r + 1) / 2 * 255)));
                g = Math.max(0, Math.min(255, Math.round((g + 1) / 2 * 255)));
                b = Math.max(0, Math.min(255, Math.round((b + 1) / 2 * 255)));
                a = Math.max(0, Math.min(255, Math.round((a + 1) / 2 * 255)));

                imgData.data[idx] = r;
                imgData.data[idx + 1] = g;
                imgData.data[idx + 2] = b;
                imgData.data[idx + 3] = a;
            }
        }
        tmpCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(tmpCanvas, ox, oy, pw * scale, ph * scale);
    }
}

function renderSinglePunk(canvas, punk, scale) {
    const pw = 24, ph = 24;
    canvas.width = pw;
    canvas.height = ph;
    canvas.style.width = (pw * scale) + 'px';
    canvas.style.height = (ph * scale) + 'px';
    canvas.style.imageRendering = 'pixelated';

    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;

    const imgData = ctx.createImageData(pw, ph);
    for (let y = 0; y < ph; y++) {
        for (let x = 0; x < pw; x++) {
            const idx = (y * pw + x) * 4;
            let r = punk[0 * ph * pw + y * pw + x];
            let g = punk[1 * ph * pw + y * pw + x];
            let b = punk[2 * ph * pw + y * pw + x];
            let a = punk[3 * ph * pw + y * pw + x];

            r = Math.max(0, Math.min(255, Math.round((r + 1) / 2 * 255)));
            g = Math.max(0, Math.min(255, Math.round((g + 1) / 2 * 255)));
            b = Math.max(0, Math.min(255, Math.round((b + 1) / 2 * 255)));
            a = Math.max(0, Math.min(255, Math.round((a + 1) / 2 * 255)));

            imgData.data[idx] = r;
            imgData.data[idx + 1] = g;
            imgData.data[idx + 2] = b;
            imgData.data[idx + 3] = a;
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function generatePunks(count, seed) {
    if (!modelWeights) return null;
    const noise = randn(seed);
    const noiseArray = new Float32Array(noise);
    const result = ganForward(noiseArray, modelWeights);
    return result.data;
}

function generateGallery() {
    if (!modelWeights) return;
    const galleryGrid = document.getElementById('galleryGrid');
    galleryGrid.innerHTML = '';
    const seeds = [11111, 22222, 33333, 44444, 55555, 66666];

    seeds.forEach((seed, i) => {
        const noise = randn(seed);
        const output = ganForward(new Float32Array(noise), modelWeights);
        const punk = output.data;

        const card = document.createElement('div');
        card.className = 'gallery-card';
        const c = document.createElement('canvas');
        renderSinglePunk(c, punk, 8);
        card.appendChild(c);
        const label = document.createElement('span');
        label.className = 'gallery-label';
        label.textContent = 'Punk #' + (i + 1);
        card.appendChild(label);
        galleryGrid.appendChild(card);
    });
}

function generateHero() {
    if (!modelWeights) return;
    const heroCanvas = document.getElementById('heroCanvas');
    const rng = mulberry32(42);
    const punks = [];
    for (let i = 0; i < 16; i++) {
        const noise = randn(Math.floor(rng() * 9999999));
        const output = ganForward(new Float32Array(noise), modelWeights);
        punks.push(output.data);
    }
    renderPunksToCanvas(heroCanvas, punks, 4, 12);

    heroCanvas.style.width = Math.min(400, window.innerWidth * 0.4) + 'px';
    heroCanvas.style.height = 'auto';
}

// --- Main application ---

document.addEventListener('DOMContentLoaded', () => {
    const btnGenerate = document.getElementById('btnGenerate');
    const btnDownload = document.getElementById('btnDownload');
    const btnReset = document.getElementById('btnReset');
    const punksCanvas = document.getElementById('punksCanvas');
    const placeholder = document.getElementById('placeholder');
    const genCount = document.getElementById('genCount');
    const genSeed = document.getElementById('genSeed');
    const genTime = document.getElementById('genTime');

    loadModel();

    btnGenerate.addEventListener('click', () => {
        if (!modelLoaded) {
            alert('El modelo GAN no esta cargado. Ejecuta "python generate_punks.py" para crear model_weights.json, luego sube todo a Vercel.');
            return;
        }

        btnGenerate.classList.add('loading-state');
        btnGenerate.innerHTML = '<span class="btn-icon">&#9889;</span> Generando...';
        btnGenerate.disabled = true;

        setTimeout(() => {
            const t0 = performance.now();
            currentSeed = Math.floor(Math.random() * 9999999);
            const rng = mulberry32(currentSeed);

            const allPunks = [];
            const batchSize = 8;
            const totalPunks = 64;

            for (let i = 0; i < totalPunks; i++) {
                const punkSeed = Math.floor(rng() * 9999999);
                const noise = randn(punkSeed);
                const output = ganForward(new Float32Array(noise), modelWeights);
                allPunks.push(output.data);
            }

            const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

            renderPunksToCanvas(punksCanvas, allPunks, 8, 10);
            punksCanvas.classList.remove('hidden');
            punksCanvas.classList.add('punk-reveal');
            placeholder.classList.add('hidden');

            btnGenerate.classList.remove('loading-state');
            btnGenerate.innerHTML = '<span class="btn-icon">&#9889;</span> Generar de Nuevo';
            btnGenerate.disabled = false;
            btnDownload.disabled = false;
            btnReset.classList.remove('hidden');

            genCount.textContent = 'Punks: 64';
            genSeed.textContent = 'Seed: ' + currentSeed;
            genTime.textContent = 'Tiempo: ' + elapsed + 's';
        }, 50);
    });

    btnDownload.addEventListener('click', () => {
        const link = document.createElement('a');
        link.href = punksCanvas.toDataURL('image/png');
        link.download = 'cryptopunks_seed_' + currentSeed + '.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    btnReset.addEventListener('click', () => {
        punksCanvas.classList.add('hidden');
        punksCanvas.classList.remove('punk-reveal');
        placeholder.classList.remove('hidden');
        btnDownload.disabled = true;
        btnReset.classList.add('hidden');
        btnGenerate.innerHTML = '<span class="btn-icon">&#9889;</span> Generar Punks';
        genCount.textContent = 'Punks: 0';
        genSeed.textContent = 'Seed: ---';
        genTime.textContent = 'Tiempo: ---';
        currentSeed = null;
    });

    document.getElementById('colabLink').addEventListener('click', (e) => {
        e.preventDefault();
        const code = encodeURIComponent(`# CryptoPunks GAN Generator\n!pip install torch torchvision huggingface_hub\nimport torch\nfrom huggingface_hub import hf_hub_download\nfrom torch import nn\nfrom torchvision.utils import save_image\nimport matplotlib.pyplot as plt\nimport matplotlib.image as mpimg\n\nclass Generator(nn.Module):\n    def __init__(self, nc=4, nz=100, ngf=64):\n        super(Generator, self).__init__()\n        self.network = nn.Sequential(\n            nn.ConvTranspose2d(nz, ngf*4, 3, 1, 0, bias=False),\n            nn.BatchNorm2d(ngf*4), nn.ReLU(True),\n            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),\n            nn.BatchNorm2d(ngf*2), nn.ReLU(True),\n            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 0, bias=False),\n            nn.BatchNorm2d(ngf), nn.ReLU(True),\n            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh(),\n        )\n    def forward(self, x):\n        return self.network(x)\n\nmodel = Generator()\nweights = hf_hub_download('nateraw/cryptopunks-gan','generator.pth')\nmodel.load_state_dict(torch.load(weights, map_location='cpu'))\nmodel.eval()\nout = model(torch.randn(64,100,1,1))\nsave_image(out,"punks.png",normalize=True)\nimg = mpimg.imread("punks.png")\nplt.figure(figsize=(10,10))\nplt.imshow(img); plt.axis("off"); plt.title("Generated CryptoPunks"); plt.show()`);
        window.open('https://colab.research.google.com/notebook#fileId=&create=true&code=' + code, '_blank');
    });

    document.getElementById('downloadScript').addEventListener('click', (e) => {
        e.preventDefault();
        const pyCode = `import torch\nfrom huggingface_hub import hf_hub_download\nfrom torch import nn\nfrom torchvision.utils import save_image\nimport matplotlib.pyplot as plt\nimport matplotlib.image as mpimg\nimport json\nimport base64\nimport numpy as np\n\n\nclass Generator(nn.Module):\n    def __init__(self, nc=4, nz=100, ngf=64):\n        super(Generator, self).__init__()\n        self.network = nn.Sequential(\n            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 0, bias=False),\n            nn.BatchNorm2d(ngf * 4),\n            nn.ReLU(True),\n            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),\n            nn.BatchNorm2d(ngf * 2),\n            nn.ReLU(True),\n            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 0, bias=False),\n            nn.BatchNorm2d(ngf),\n            nn.ReLU(True),\n            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),\n            nn.Tanh(),\n        )\n\n    def forward(self, input):\n        output = self.network(input)\n        return output\n\n\nmodel = Generator()\nweights_path = hf_hub_download('nateraw/cryptopunks-gan', 'generator.pth')\nmodel.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))\nmodel.eval()\n\nout = model(torch.randn(64, 100, 1, 1))\nsave_image(out, "punks.png", normalize=True)\nprint("Generated punks.png with 64 punks!")\n\nimg = mpimg.imread("punks.png")\nplt.imshow(img)\nplt.axis("off")\nplt.title("Generated CryptoPunks")\nplt.show()\n\nprint("Exporting model weights for web...")\nweights = {}\nfor name, param in model.named_parameters():\n    arr = param.detach().cpu().numpy()\n    weights[name] = {\n        'shape': list(arr.shape),\n        'dtype': str(arr.dtype),\n        'data': base64.b64encode(arr.tobytes()).decode('ascii')\n    }\nfor name, buf in model.named_buffers():\n    arr = buf.detach().cpu().numpy()\n    weights[name] = {\n        'shape': list(arr.shape),\n        'dtype': str(arr.dtype),\n        'data': base64.b64encode(arr.tobytes()).decode('ascii')\n    }\n\nwith open('model_weights.json', 'w') as f:\n    json.dump(weights, f)\n\nprint("Created model_weights.json - upload this file with your web page!")\n`;
        const blob = new Blob([pyCode], { type: 'text/plain' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'generate_punks.py';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    const observerOptions = { threshold: 0.1, rootMargin: '0px 0px -50px 0px' };
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    document.querySelectorAll('.about-card, .resource-card').forEach((el, i) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease ' + (i * 0.1) + 's, transform 0.6s ease ' + (i * 0.1) + 's';
        observer.observe(el);
    });

    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        navbar.style.borderBottomColor = window.scrollY > 50 ? 'rgba(108,92,231,0.2)' : 'var(--border-subtle)';
    });

    document.querySelectorAll('.stat-number').forEach(stat => {
        const n = parseInt(stat.textContent);
        if (isNaN(n)) return;
        let c = 0;
        const inc = Math.ceil(n / 30);
        const timer = setInterval(() => {
            c += inc;
            if (c >= n) { c = n; clearInterval(timer); }
            stat.textContent = c;
        }, 40);
    });
});