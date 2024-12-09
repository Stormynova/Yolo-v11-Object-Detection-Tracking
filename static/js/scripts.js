function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', newTheme);
}

function captureFrame() {
    fetch('/detect', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        const detectionList = document.getElementById('detection-list');
        detectionList.innerHTML = '';
        
        if (data.error) {
            detectionList.innerHTML = `<div class="detection-item">Error: ${data.error}</div>`;
            return;
        }

        data.detections.forEach(detection => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            
            const nameSpan = document.createElement('span');
            nameSpan.textContent = detection.class_name;
            
            const confidenceBadge = document.createElement('span');
            confidenceBadge.className = 'confidence-badge';
            confidenceBadge.textContent = `${(detection.confidence * 100).toFixed(1)}%`;
            
            item.appendChild(nameSpan);
            item.appendChild(confidenceBadge);
            detectionList.appendChild(item);
        });
    })
    .catch(error => {
        console.error('Error:', error);
        const detectionList = document.getElementById('detection-list');
        detectionList.innerHTML = '<div class="detection-item">Error: Failed to detect objects</div>';
    });
}

function updateDetectionList(detections) {
    const detectionList = document.getElementById('detection-list');
    detectionList.innerHTML = '';
    
    detections.forEach(detection => {
        const item = document.createElement('div');
        item.className = 'detection-item';
        
        // Header with name and confidence
        const header = document.createElement('div');
        header.className = 'detection-header';
        
        const nameSpan = document.createElement('span');
        nameSpan.innerHTML = `<i class="fas fa-tag"></i>${detection.class_name}`;
        
        const confidenceBadge = document.createElement('span');
        confidenceBadge.className = 'confidence-badge';
        confidenceBadge.innerHTML = `<i class="fas fa-percentage"></i>${(detection.confidence * 100).toFixed(1)}%`;
        
        header.appendChild(nameSpan);
        header.appendChild(confidenceBadge);
        
        // Metrics section
        const metrics = document.createElement('div');
        metrics.className = 'detection-metrics';
        
        // Distance bar
        const distanceRow = document.createElement('div');
        distanceRow.className = 'metric-row';
        
        const distanceLabel = document.createElement('span');
        distanceLabel.className = 'metric-label';
        distanceLabel.innerHTML = `<i class="fas fa-ruler"></i> Distance`;
        
        const distanceBarContainer = document.createElement('div');
        distanceBarContainer.className = 'distance-bar-container';
        
        const distanceBar = document.createElement('div');
        distanceBar.className = 'distance-bar';
        
        // Add distance value text
        const distanceText = document.createElement('span');
        distanceText.className = 'metric-value';
        distanceText.textContent = detection.distance ? `${detection.distance}m` : 'N/A';
        
        const distancePercent = Math.min((detection.distance || 0) / 5 * 100, 100);
        distanceBar.style.width = `${distancePercent}%`;
        
        distanceBarContainer.appendChild(distanceBar);
        distanceRow.appendChild(distanceLabel);
        distanceRow.appendChild(distanceBarContainer);
        distanceRow.appendChild(distanceText);
        
        // Angle indicator
        const angleRow = document.createElement('div');
        angleRow.className = 'metric-row';
        
        const angleLabel = document.createElement('span');
        angleLabel.className = 'metric-label';
        angleLabel.innerHTML = `<i class="fas fa-compass"></i> Angle`;
        
        const angleIndicator = document.createElement('div');
        angleIndicator.className = 'angle-indicator';
        
        const centerMarker = document.createElement('div');
        centerMarker.className = 'angle-marker';
        
        const anglePointer = document.createElement('div');
        anglePointer.className = 'angle-pointer';
        // Convert angle (-45 to 45 degrees) to position (0 to 100%)
        const anglePercent = ((detection.angle || 0) + 45) / 90 * 100;
        anglePointer.style.left = `${anglePercent}%`;
        
        angleIndicator.appendChild(centerMarker);
        angleIndicator.appendChild(anglePointer);
        
        // Add angle value text
        const angleText = document.createElement('span');
        angleText.className = 'metric-value';
        angleText.textContent = detection.angle ? `${detection.angle}°` : 'N/A';
        
        angleRow.appendChild(angleLabel);
        angleRow.appendChild(angleIndicator);
        angleRow.appendChild(angleText);
        
        metrics.appendChild(distanceRow);
        metrics.appendChild(angleRow);
        
        item.appendChild(header);
        item.appendChild(metrics);
        detectionList.appendChild(item);
    });
}

// Add this function to periodically update detections
function startRealTimeUpdates() {
    setInterval(() => {
        fetch('/detect', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                // Filter detections based on current threshold
                const filteredDetections = data.detections.filter(detection => 
                    detection.confidence * 100 >= currentThreshold
                );
                updateDetectionList(filteredDetections);
            }
        })
        .catch(error => console.error('Error:', error));
    }, 1000); // Update every second
}

// Call this when the page loads
document.addEventListener('DOMContentLoaded', startRealTimeUpdates); 

let currentThreshold = 50; // Default threshold value (matches HTML default)

function updateThreshold(value) {
    currentThreshold = parseInt(value);
    document.getElementById('threshold-value').textContent = `${value}%`;
    
    fetch(`/update_threshold/${value}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Trigger a new detection to update the display
            captureFrame();
        }
    })
    .catch(error => console.error('Error updating threshold:', error));
}

function checkCamera() {
    fetch('/camera/check')
        .then(response => response.json())
        .then(data => {
            if (!data.available) {
                showCameraError(data.message);
            } else {
                hideCameraError();
            }
        })
        .catch(error => {
            showCameraError('Failed to connect to camera');
            console.error('Camera check error:', error);
        });
}

function showCameraError(message) {
    const videoFeed = document.getElementById('video-feed');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'camera-error';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <p>${message}</p>
        <p>Please ensure your camera is connected and you've granted camera permissions.</p>
    `;
    videoFeed.parentNode.insertBefore(errorDiv, videoFeed);
    videoFeed.style.display = 'none';
}

function hideCameraError() {
    const errorDiv = document.querySelector('.camera-error');
    if (errorDiv) {
        errorDiv.remove();
    }
    document.getElementById('video-feed').style.display = 'block';
}

// Check camera status periodically
setInterval(checkCamera, 5000); 