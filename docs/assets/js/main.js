// Mobile Navigation Toggle
const navToggle = document.querySelector('.nav-toggle');
const navMenu = document.querySelector('.nav-menu');

if (navToggle) {
    navToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        navToggle.classList.toggle('active');
    });
}

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-menu a').forEach(link => {
    link.addEventListener('click', () => {
        navMenu.classList.remove('active');
        navToggle.classList.remove('active');
    });
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const headerOffset = 80;
            const elementPosition = target.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Navbar background on scroll
const navbar = document.querySelector('.navbar');
let lastScrollY = window.scrollY;

window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(15, 23, 42, 0.95)';
    } else {
        navbar.style.background = 'rgba(15, 23, 42, 0.9)';
    }
    lastScrollY = window.scrollY;
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
        }
    });
}, observerOptions);

// Observe elements for animation
document.querySelectorAll('.feature-card, .problem-card, .module-card, .tech-card, .step').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
});

// Add animation class style
const style = document.createElement('style');
style.textContent = `
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
`;
document.head.appendChild(style);

// Memory diagram animation
const memoryLayers = document.querySelectorAll('.memory-layer');
let currentLayer = 0;

function animateMemoryLayers() {
    memoryLayers.forEach((layer, index) => {
        layer.style.transform = index === currentLayer ? 'scale(1.05)' : 'scale(1)';
    });
    currentLayer = (currentLayer + 1) % memoryLayers.length;
}

// Start animation after page load
if (memoryLayers.length > 0) {
    setInterval(animateMemoryLayers, 2000);
}

// Copy code blocks on click
document.querySelectorAll('pre code').forEach(block => {
    block.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(block.textContent);

            // Show feedback
            const feedback = document.createElement('div');
            feedback.textContent = 'Copied!';
            feedback.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: #22c55e;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                z-index: 9999;
                animation: fadeOut 2s forwards;
            `;
            document.body.appendChild(feedback);

            setTimeout(() => feedback.remove(), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    });

    block.style.cursor = 'pointer';
    block.title = 'Click to copy';
});

// Add fadeOut animation
const fadeOutStyle = document.createElement('style');
fadeOutStyle.textContent = `
    @keyframes fadeOut {
        0% { opacity: 1; }
        70% { opacity: 1; }
        100% { opacity: 0; }
    }
`;
document.head.appendChild(fadeOutStyle);

// Console welcome message
console.log('%c🧠 MemoryForge', 'font-size: 24px; font-weight: bold; color: #6366f1;');
console.log('%cHierarchical Context Memory System', 'font-size: 14px; color: #94a3b8;');
console.log('%chttps://github.com/GeoffreyWang1117/MemoryForge', 'font-size: 12px; color: #818cf8;');
