
// Enhanced Dashboard JavaScript with Excel Import Support
console.log('Enhanced Dashboard.js loading with Excel import features...');

// Inherit all original dashboard functionality
// Enhanced features for data import

function showDataImportModal() {
    // Redirect to data import page
    window.location.href = '/data-import';
}

function handleFileUpload(inputId, dataType) {
    const input = document.getElementById(inputId);
    const file = input.files[0];
    
    if (!file) {
        showNotification('Please select a file', 'warning');
        return;
    }
    
    // Validate file type
    const validTypes = ['.xlsx', '.xls', '.csv'];
    const isValid = validTypes.some(type => file.name.toLowerCase().endsWith(type));
    
    if (!isValid) {
        showNotification('Please select an Excel or CSV file', 'error');
        return;
    }
    
    // Show upload progress
    showNotification(`Uploading ${dataType} data...`, 'info');
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch(`/upload-${dataType}-data`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification(`Successfully imported ${data.records_imported} records`, 'success');
            setTimeout(() => location.reload(), 2000);
        } else {
            showNotification(data.message || 'Upload failed', 'error');
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        showNotification('Upload failed: ' + error.message, 'error');
    });
}

// Add enhanced notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type === 'warning' ? 'warning' : type === 'success' ? 'success' : 'info'}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        max-width: 400px;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    `;
    notification.innerHTML = `
        <strong>${type.charAt(0).toUpperCase() + type.slice(1)}:</strong> ${message}
        <button type="button" class="btn-close float-end" onclick="this.parentElement.remove()"></button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

console.log('Enhanced Dashboard.js loaded successfully!');
