document.addEventListener('DOMContentLoaded', function() {
    // Initialize Vanta.js 3D background
    if (window.VANTA && document.getElementById('vanta-background')) {
        VANTA.WAVES({
            el: "#vanta-background",
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200.00,
            minWidth: 200.00,
            scale: 1.00,
            scaleMobile: 1.00,
            color: 0x4361ee,
            shininess: 50.00,
            waveHeight: 15.00,
            waveSpeed: 0.50,
            zoom: 0.8
        });
    }

    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }

    // Mobile menu toggle
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', function() {
            navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
        });
    }

    // Animate elements on scroll
    const animateElements = document.querySelectorAll('.animate-step');
    if (animateElements.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animated');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1
        });

        animateElements.forEach(element => {
            observer.observe(element);
        });
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
                
                // Close mobile menu if open
                if (navLinks && navLinks.style.display === 'flex') {
                    navLinks.style.display = 'none';
                }
            }
        });
    });

    // Handle responsive menu on resize
    if (navLinks) {
        window.addEventListener('resize', function() {
            if (window.innerWidth > 992) {
                navLinks.style.display = 'flex';
            } else {
                navLinks.style.display = 'none';
            }
        });
    }

    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const nextBtn = document.getElementById('next-btn');
    const videoPreview = document.getElementById('video-preview');
    const answerInput = document.getElementById('answer-text');
    const timerDisplay = document.querySelector('.timer');
    
    let mediaRecorder;
    let recordedChunks = [];
    let timerInterval;
    let seconds = 0;
    let speechRecognition;
    
    // Check for browser support
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
        alert('Your browser does not support video recording. Please use Chrome or Firefox.');
        return;
    }
    
    // Initialize camera
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
            videoPreview.srcObject = stream;
            
            // Initialize media recorder
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = function(e) {
                if (e.data.size > 0) {
                    recordedChunks.push(e.data);
                }
            };
            
            mediaRecorder.onstop = function() {
                // In a real app, you would upload the video blob to your server
                // For demo, we'll just simulate processing time
                simulateProcessing();
            };
        })
        .catch(err => {
            console.error('Error accessing media devices:', err);
            alert('Could not access camera and microphone. Please ensure you have granted permissions.');
        });
    
    // Start recording
    startBtn.addEventListener('click', function() {
        recordedChunks = [];
        seconds = 0;
        updateTimer();
        
        mediaRecorder.start();
        startSpeechRecognition();
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        timerInterval = setInterval(updateTimer, 1000);
    });
    
    // Stop recording
    stopBtn.addEventListener('click', function() {
        mediaRecorder.stop();
        stopSpeechRecognition();
        
        clearInterval(timerInterval);
        stopBtn.disabled = true;
    });
    
    // Update timer display
    function updateTimer() {
        seconds++;
        const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
        const secs = (seconds % 60).toString().padStart(2, '0');
        timerDisplay.textContent = `${mins}:${secs}`;
    }
    
    // Add to your existing interview.js
    function analyzeSpeech(text) {
        // Simulate real-time analysis (in production, use actual NLP)
        const clarity = Math.min(100, Math.floor(Math.random() * 30) + 70);
        const relevance = Math.min(100, Math.floor(Math.random() * 20) + 75);
        const depth = Math.min(100, Math.floor(Math.random() * 25) + 65);
        
        // Update UI
        document.getElementById('clarity-bar').style.width = `${clarity}%`;
        document.getElementById('relevance-bar').style.width = `${relevance}%`;
        document.getElementById('depth-bar').style.width = `${depth}%`;
        
        return { clarity, relevance, depth };
    }

    // Start speech recognition
    function startSpeechRecognition() {
        if ('webkitSpeechRecognition' in window) {
            speechRecognition = new webkitSpeechRecognition();
            speechRecognition.continuous = true;
            speechRecognition.interimResults = true;
            
            // Modify the speech recognition callback
            speechRecognition.onresult = function(event) {
                let interimTranscript = '';
                let finalTranscript = '';
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                        analyzeSpeech(transcript);  // Analyze each final transcript
                    } else {
                        interimTranscript += transcript;
                    }
                }
                
                answerInput.value = finalTranscript || interimTranscript;
            };
            
            speechRecognition.start();
        } else {
            console.warn('Speech recognition not supported');
            // Fallback to manual input
            const manualInput = document.createElement('textarea');
            manualInput.placeholder = 'Type your answer here...';
            document.querySelector('.question-container').appendChild(manualInput);
            
            answerInput.value = 'Speech-to-text not supported - using manual input';
        }
    }
    
    // Stop speech recognition
    function stopSpeechRecognition() {
        if (speechRecognition) {
            speechRecognition.stop();
        }
    }
    
    // Simulate processing
    function simulateProcessing() {
        nextBtn.disabled = true;
        nextBtn.textContent = 'Processing...';
        
        setTimeout(() => {
            nextBtn.disabled = false;
            nextBtn.style.display = 'inline-block';
            nextBtn.textContent = 'Next Question';
        }, 2000);
    }
});