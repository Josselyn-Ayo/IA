let dataLoaded = false;
let currentBatchIndex = -1;
let currentBatchKey = null;
let batchKeys = [];

document.addEventListener('DOMContentLoaded', () => {
    const btnGenerate = document.getElementById('btnGenerate');
    const btnDownload = document.getElementById('btnDownload');
    const btnReset = document.getElementById('btnReset');
    const punksImage = document.getElementById('punksImage');
    const placeholder = document.getElementById('placeholder');
    const placeholderMsg = document.getElementById('placeholderMsg');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const genBatch = document.getElementById('genBatch');
    const genCount = document.getElementById('genCount');
    const genMode = document.getElementById('genMode');
    const heroPreview = document.getElementById('heroPreview');
    const heroText = document.getElementById('heroText');

    if (typeof PUNKS_DATA !== 'undefined' && PUNKS_DATA) {
        batchKeys = Object.keys(PUNKS_DATA);
        const totalBatches = typeof PUNKS_TOTAL_BATCHES !== 'undefined' ? PUNKS_TOTAL_BATCHES : batchKeys.length;
        dataLoaded = true;

        statusDot.className = 'status-indicator ready';
        statusText.textContent = 'Modelo GAN cargado - ' + (totalBatches * 64) + ' punks disponibles';
        btnGenerate.disabled = false;
        btnGenerate.innerHTML = '<span class="btn-icon">&#9654;</span> Generar Punks';
        placeholderMsg.textContent = 'Presiona "Generar Punks" para crear tu coleccion';
        genMode.textContent = 'Modo: GAN Real';
        document.getElementById('statBatches').textContent = totalBatches;

        // Load hero image
        const heroImg = new Image();
        heroImg.onload = function() {
            heroPreview.innerHTML = '';
            heroImg.style.imageRendering = 'pixelated';
            heroPreview.appendChild(heroImg);
        };
        heroImg.src = 'data:image/png;base64,' + PUNKS_DATA[batchKeys[0]];
    } else {
        statusDot.className = 'status-indicator error';
        statusText.textContent = 'Sin datos - Ejecuta python generate_punks.py';
        btnGenerate.disabled = false;
        btnGenerate.innerHTML = '<span class="btn-icon">&#9654;</span> Generar Punks';
        placeholderMsg.innerHTML = 'Ejecuta <code>python generate_punks.py</code> para generar los datos del modelo.<br>Luego abre esta pagina con un servidor web o subela a Vercel.';
        placeholderMsg.classList.add('error-msg');
        genMode.textContent = 'Modo: Sin datos';
        heroText.textContent = 'Ejecuta python generate_punks.py';
    }

    btnGenerate.addEventListener('click', () => {
        if (!dataLoaded) {
            alert('Datos del modelo no cargados.\n\nPasos:\n1. Ejecuta: python generate_punks.py\n2. Esto creara punks_data.js\n3. Abre index.html con un servidor web o sube a Vercel\n\nLos punks que veras son generados por la misma red GAN de tu codigo Python.');
            return;
        }

        btnGenerate.classList.add('loading-state');
        btnGenerate.innerHTML = '<span class="btn-icon">&#9654;</span> Cargando...';
        btnGenerate.disabled = true;

        setTimeout(() => {
            let newIndex;
            do {
                newIndex = Math.floor(Math.random() * batchKeys.length);
            } while (newIndex === currentBatchIndex && batchKeys.length > 1);
            currentBatchIndex = newIndex;
            currentBatchKey = batchKeys[newIndex];

            punksImage.onload = function() {
                punksImage.classList.remove('hidden');
                punksImage.classList.add('punk-reveal');
                placeholder.classList.add('hidden');

                btnGenerate.classList.remove('loading-state');
                btnGenerate.innerHTML = '<span class="btn-icon">&#9654;</span> Generar de Nuevo';
                btnGenerate.disabled = false;
                btnDownload.disabled = false;
                btnReset.classList.remove('hidden');

                genBatch.textContent = 'Lote: ' + (currentBatchIndex + 1) + '/' + batchKeys.length;
                genCount.textContent = 'Punks: 64';
            };

            punksImage.src = 'data:image/png;base64,' + PUNKS_DATA[currentBatchKey];
        }, 200);
    });

    btnDownload.addEventListener('click', () => {
        if (!currentBatchKey) return;
        const link = document.createElement('a');
        link.href = 'data:image/png;base64,' + PUNKS_DATA[currentBatchKey];
        link.download = 'cryptopunks_batch_' + (currentBatchIndex + 1) + '.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    btnReset.addEventListener('click', () => {
        punksImage.classList.add('hidden');
        punksImage.classList.remove('punk-reveal');
        placeholder.classList.remove('hidden');
        btnDownload.disabled = true;
        btnReset.classList.add('hidden');
        btnGenerate.innerHTML = '<span class="btn-icon">&#9654;</span> Generar Punks';
        genBatch.textContent = 'Lote: --';
        genCount.textContent = 'Punks: 0';
        currentBatchIndex = -1;
        currentBatchKey = null;
    });

    document.getElementById('downloadScript').addEventListener('click', (e) => {
        e.preventDefault();
        const pyCode = `import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import base64
import os


class Generator(nn.Module):
    def __init__(self, nc=4, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.network(input)
        return output


model = Generator()
weights_path = hf_hub_download('nateraw/cryptopunks-gan', 'generator.pth')
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

print("Generando punks.png...")
out = model(torch.randn(64, 100, 1, 1))
save_image(out, "punks.png", normalize=True)
print("Generated punks.png with 64 punks!")

img = mpimg.imread("punks.png")
plt.imshow(img)
plt.axis("off")
plt.title("Generated CryptoPunks")
plt.show()

print("Generando lotes para la pagina web...")
NUM_BATCHES = 10
batches = {}

for i in range(NUM_BATCHES):
    noise = torch.randn(64, 100, 1, 1)
    with torch.no_grad():
        output = model(noise)

    filename = f"punks_{i}.png"
    save_image(output, filename, normalize=True)

    with open(filename, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("ascii")

    batches[f"batch_{i}"] = img_data
    print(f"  Lote {i+1}/{NUM_BATCHES} generado")

    os.remove(filename)

js_content = "// CryptoPunks GAN - Pre-generated batches\\n"
js_content += "// Generated by generate_punks.py using the real GAN model\\n"
js_content += "// Each batch contains 64 punks as base64 PNG images\\n\\n"
js_content += "const PUNKS_DATA = " + json.dumps(batches) + ";\\n"
js_content += "const PUNKS_TOTAL_BATCHES = " + str(NUM_BATCHES) + ";\\n"

with open("punks_data.js", "w", encoding="utf-8") as f:
    f.write(js_content)

print(f"\\nHecho! Se creo punks_data.js con {NUM_BATCHES} lotes ({NUM_BATCHES * 64} punks)")
print("Sube index.html, styles.css, script.js y punks_data.js a Vercel")
`;
        const blob = new Blob([pyCode], { type: 'text/plain' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'generate_punks.py';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    document.getElementById('btnCopy').addEventListener('click', () => {
        const code = document.querySelector('.code-block code');
        navigator.clipboard.writeText(code.textContent).then(() => {
            const btn = document.getElementById('btnCopy');
            btn.textContent = '\u2705 Copiado!';
            setTimeout(() => { btn.textContent = 'Copiar'; }, 2000);
        });
    });

    // Scroll animations
    const scrollObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.scroll-animate').forEach((el, i) => {
        el.style.transitionDelay = (i * 0.08) + 's';
        scrollObserver.observe(el);
    });

    // Particles
    const particlesContainer = document.getElementById('particles');
    for (let i = 0; i < 30; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = Math.random() * 100 + '%';
        p.style.animationDuration = (8 + Math.random() * 15) + 's';
        p.style.animationDelay = (Math.random() * 10) + 's';
        p.style.width = (2 + Math.random() * 4) + 'px';
        p.style.height = p.style.width;
        const colors = ['var(--accent-primary)', 'var(--accent-secondary)', 'var(--accent-tertiary)'];
        p.style.background = colors[Math.floor(Math.random() * colors.length)];
        particlesContainer.appendChild(p);
    }

    // Navbar scroll effect
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.borderBottomColor = 'rgba(108,92,231,0.3)';
            navbar.style.background = 'rgba(10,10,15,0.95)';
        } else {
            navbar.style.borderBottomColor = 'var(--border-subtle)';
            navbar.style.background = 'rgba(10,10,15,0.85)';
        }
    });

    // Counter animation
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