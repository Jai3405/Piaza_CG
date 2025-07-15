import React, { useState } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ResultsDisplay from './components/ResultsDisplay';

function App() {
  const [extractedData, setExtractedData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setExtractedData(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch('http://localhost:5000/extract', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Failed to extract data.');
      }
      const data = await response.json();
      setExtractedData(data.extracted_data || data.extractedData || data);
    } catch (err: any) {
      setError(err.message || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>OlmOCR Data Extraction</h1>
      <FileUpload onFileUpload={handleFileUpload} loading={loading} />
      {error && <div className="error-message">{error}</div>}
      <ResultsDisplay data={extractedData} loading={loading} />
    </div>
  );
}

export default App;
