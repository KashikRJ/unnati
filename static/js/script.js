function displayImage(input) {
    const preview = document.getElementById('image-preview');
    const result = document.getElementById('result');
    result.textContent = ''; // Clear previous result
    preview.innerHTML = ''; // Clear previous image
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Selected Image" style="width: 100%; border-radius: 10px;"/>`;
        }
        reader.readAsDataURL(input.files[0]);
    }
}

async function submitForm() {
    const form = document.getElementById('upload-form');
    const formData = new FormData(form);
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    if (response.ok) {
        const result = await response.json();
        document.getElementById('result').textContent = `Prediction: ${result.prediction}`;
    } else {
        document.getElementById('result').textContent = 'An error occurred. Please try again.';
    }
}

function toggleTheme() {
    document.body.classList.toggle('dark-mode');
}
