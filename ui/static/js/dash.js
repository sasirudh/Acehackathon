class BouncyBlockClock {
    constructor(qs) {
        this.el = document.querySelector(qs);
        this.time = { a: [], b: [] };
        this.rollClass = "clock__block--bounce";
        this.digitsTimeout = null;
        this.rollTimeout = null;
        this.mod = 0 * 60 * 1000;
        this.loop();
    }

    animateDigits() {
        const groups = this.el.querySelectorAll("[data-time-group]");
        Array.from(groups).forEach((group,i) => {
            const { a, b } = this.time;
            if (a[i] !== b[i]) group.classList.add(this.rollClass);
        });
        clearTimeout(this.rollTimeout);
        this.rollTimeout = setTimeout(this.removeAnimations.bind(this),900);
    }

    displayTime() {
        const timeDigits = [...this.time.b];
        const ap = timeDigits.pop();
        this.el.ariaLabel = `${timeDigits.join(":")} ${ap}`;

        Object.keys(this.time).forEach(letter => {
            const letterEls = this.el.querySelectorAll(`[data-time="${letter}"]`);
            Array.from(letterEls).forEach((el,i) => {
                el.textContent = this.time[letter][i];
            });
        });
    }

    loop() {
        this.updateTime();
        this.displayTime();
        this.animateDigits();
        this.tick();
    }

    removeAnimations() {
        const groups = this.el.querySelectorAll("[data-time-group]");
        Array.from(groups).forEach(group => {
            group.classList.remove(this.rollClass);
        });
    }

    tick() {
        clearTimeout(this.digitsTimeout);
        this.digitsTimeout = setTimeout(this.loop.bind(this),1e3);    
    }

    updateTime() {
        const rawDate = new Date();
        const date = new Date(Math.ceil(rawDate.getTime() / 1e3) * 1e3 + this.mod);
        let h = date.getHours();
        const m = date.getMinutes();
        const s = date.getSeconds();
        const ap = h < 12 ? "AM" : "PM";

        if (h === 0) h = 12;
        if (h > 12) h -= 12;

        this.time.a = [...this.time.b];
        this.time.b = [
            (h < 10 ? `0${h}` : `${h}`),
            (m < 10 ? `0${m}` : `${m}`),
            (s < 10 ? `0${s}` : `${s}`),
            ap
        ];

        if (!this.time.a.length) this.time.a = [...this.time.b];
    }
}

document.addEventListener("DOMContentLoaded", () => {
    // Initialize clock
    const clock = new BouncyBlockClock(".clock");
    
    // Get DOM elements
    const terminal = document.getElementById('terminal');
    const alertBox = document.getElementById('alert-box');
    const loadingBar = document.getElementById('loading-bar');
    const fileUpload = document.getElementById('file-upload');
    const videoFrame = document.getElementById('video-frame');
    
    // Create fuel display element
    const fuelDisplay = document.createElement('div');
    fuelDisplay.className = 'fuel-display';
    fuelDisplay.innerHTML = 'Total Fuel Injected: 0.00 L';
    document.querySelector('.right-column').insertBefore(fuelDisplay, alertBox.nextSibling);
    
    // System heartbeat
    setInterval(() => {
        terminal.innerHTML += `> ${new Date().toLocaleTimeString()}: System active<br>`;
        terminal.scrollTop = terminal.scrollHeight;
    }, 5000);

    // File upload handler
    fileUpload.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Show loading state
        loadingBar.style.display = 'block';
        loadingBar.querySelector('.progress').style.width = '0%';
        terminal.innerHTML += '> Starting video processing...<br>';
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            // Start upload
            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (uploadResponse.ok) {
                terminal.innerHTML += '> Video uploaded successfully<br>';
                startStatusUpdates();
            }
        } catch (error) {
            terminal.innerHTML += `> Upload failed: ${error}<br>`;
            loadingBar.style.display = 'none';
        }
    });
    
    // Status update function
    function startStatusUpdates() {
        // First update immediately
        updateStatus();
        
        // Then update every second
        const statusInterval = setInterval(updateStatus, 1000);
        
        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                // Update terminal with latest log
                if (data.logs?.length > 0) {
                    terminal.innerHTML += `> ${data.logs.slice(-1)[0]}<br>`;
                }
                
                // Update alerts
                if (data.alerts?.length > 0) {
                    alertBox.innerHTML = `<strong>ALERTS:</strong> ${data.alerts.join(', ')}`;
                    alertBox.style.borderLeftColor = '#ff0000';
                } else {
                    alertBox.innerHTML = '<strong>Status:</strong> All systems normal';
                    alertBox.style.borderLeftColor = '#00aaff';
                }
                
                // Update fuel display
                if (data.fuel !== undefined) {
                    fuelDisplay.innerHTML = `Total Fuel Injected: ${data.fuel.toFixed(2)} L`;
                }
                
                // Update loading progress
                if (data.progress !== undefined) {
                    const progress = loadingBar.querySelector('.progress');
                    progress.style.width = `${data.progress}%`;
                    progress.style.animation = 'none';
                }
                
                // Handle completion
                if (data.status === 'complete') {
                    clearInterval(statusInterval);
                    loadingBar.style.display = 'none';
                    terminal.innerHTML += '> Processing complete!<br>';
                    
                    // Show processed video
                    const video = document.createElement('video');
                    video.controls = true;
                    video.src = '/processed/output.mp4';
                    videoFrame.innerHTML = '';
                    videoFrame.appendChild(video);
                }
                
                // Handle errors
                if (data.status === 'error') {
                    clearInterval(statusInterval);
                    loadingBar.style.display = 'none';
                    terminal.innerHTML += `> ERROR: ${data.message}<br>`;
                }
                
                terminal.scrollTop = terminal.scrollHeight;
            } catch (error) {
                console.error('Status update error:', error);
            }
        }
    }
});