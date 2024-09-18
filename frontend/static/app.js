document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);  // The file uploaded by the user

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {  // FastAPI predict endpoint
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();  // Get the prediction response
        document.getElementById('transcript').textContent = `Prediction: ${result.prediction}`;
    } catch (error) {
        document.getElementById('transcript').textContent = 'Error during prediction!';
        console.error('Error:', error);
    }
});
