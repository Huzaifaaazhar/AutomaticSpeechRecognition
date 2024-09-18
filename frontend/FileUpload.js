import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
    const [file, setFile] = useState(null);
    const [prediction, setPrediction] = useState('');

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) {
            alert('Please select a file');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:8000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error('Error uploading file', error);
        }
    };

    return (
        <div className="container">
            <h1>ASR Model Prediction</h1>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload and Predict</button>
            {prediction && <div className="prediction">Prediction: {prediction}</div>}
        </div>
    );
};

export default FileUpload;
