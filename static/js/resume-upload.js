document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('resume');
    const fileNameDisplay = document.querySelector('.file-name');
    
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileNameDisplay.textContent = this.files[0].name;
        } else {
            fileNameDisplay.textContent = 'No file chosen';
        }
    });
});