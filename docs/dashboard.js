// Sample data
const sampleData = {
    samples: [
        {location: "Site A", latitude: 40.7128, longitude: -74.0060, concentration: 120, media_type: "water", date: "2024-01-01", presence: 1, polymer_type: "PE", temperature: 20.0, ph: 7.0, sample_depth: 1.0, sample_mass: 100.0},
        {location: "Site B", latitude: 34.0522, longitude: -118.2437, concentration: 85, media_type: "water", date: "2024-01-02", presence: 1, polymer_type: "PP", temperature: 22.0, ph: 7.2, sample_depth: 1.5, sample_mass: 120.0},
        {location: "Site C", latitude: 37.7749, longitude: -122.4194, concentration: 150, media_type: "soil", date: "2024-01-03", presence: 1, polymer_type: "PS", temperature: 18.0, ph: 6.8, sample_depth: 0.5, sample_mass: 200.0},
        {location: "Site D", latitude: 25.7617, longitude: -80.1918, concentration: 95, media_type: "water", date: "2024-01-04", presence: 1, polymer_type: "PET", temperature: 25.0, ph: 7.5, sample_depth: 2.0, sample_mass: 150.0},
        {location: "Site E", latitude: 47.6062, longitude: -122.3321, concentration: 110, media_type: "blood", date: "2024-01-05", presence: 1, polymer_type: "PVC", temperature: 37.0, ph: 7.4, sample_depth: 0.0, sample_mass: 50.0},
        {location: "Site F", latitude: 29.7604, longitude: -95.3698, concentration: 135, media_type: "water", date: "2024-01-06", presence: 1, polymer_type: "PE", temperature: 23.0, ph: 7.1, sample_depth: 1.2, sample_mass: 110.0},
        {location: "Site G", latitude: 39.9526, longitude: -75.1652, concentration: 75, media_type: "soil", date: "2024-01-07", presence: 0, polymer_type: null, temperature: 15.0, ph: 6.5, sample_depth: 0.8, sample_mass: 180.0},
        {location: "Site H", latitude: 33.4484, longitude: -112.0740, concentration: 200, media_type: "water", date: "2024-01-08", presence: 1, polymer_type: "PP", temperature: 28.0, ph: 7.8, sample_depth: 1.8, sample_mass: 130.0},
        {location: "Site I", latitude: 32.7767, longitude: -96.7970, concentration: 90, media_type: "blood", date: "2024-01-09", presence: 1, polymer_type: "PS", temperature: 36.5, ph: 7.3, sample_depth: 0.0, sample_mass: 45.0},
        {location: "Site J", latitude: 41.8781, longitude: -87.6298, concentration: 160, media_type: "soil", date: "2024-01-10", presence: 1, polymer_type: "PET", temperature: 12.0, ph: 6.2, sample_depth: 0.6, sample_mass: 220.0}
    ],
    timeSeries: [
        {date: "2024-01-01", "Site A": 120, "Site B": 85, "Site C": 150, "Site D": 95, "Site E": 110, "Site F": 135, "Site G": 75, "Site H": 200, "Site I": 90, "Site J": 160},
        {date: "2024-01-02", "Site A": 125, "Site B": 88, "Site C": 155, "Site D": 98, "Site E": 115, "Site F": 140, "Site G": 78, "Site H": 205, "Site I": 92, "Site J": 165},
        {date: "2024-01-03", "Site A": 118, "Site B": 82, "Site C": 148, "Site D": 93, "Site E": 108, "Site F": 132, "Site G": 72, "Site H": 198, "Site I": 88, "Site J": 158},
        {date: "2024-01-04", "Site A": 122, "Site B": 90, "Site C": 152, "Site D": 96, "Site E": 112, "Site F": 138, "Site G": 76, "Site H": 202, "Site I": 91, "Site J": 162},
        {date: "2024-01-05", "Site A": 120, "Site B": 85, "Site C": 150, "Site D": 95, "Site E": 110, "Site F": 135, "Site G": 75, "Site H": 200, "Site I": 90, "Site J": 160}
    ],
    modelMetrics: {
        tabular: {accuracy: 0.92, precision: 0.89, recall: 0.94, f1: 0.91, confusion: [[450, 30], [25, 495]], features: [
            {feature: "sample_depth", importance: 0.35},
            {feature: "latitude", importance: 0.25},
            {feature: "longitude", importance: 0.20},
            {feature: "temperature", importance: 0.12},
            {feature: "ph", importance: 0.08}
        ]},
        spectra: {accuracy: 0.88, precision: 0.85, recall: 0.90, f1: 0.87, confusion: [[420, 40], [35, 505]]},
        image: {accuracy: 0.91, precision: 0.88, recall: 0.93, f1: 0.90, confusion: [[435, 25], [30, 510]]},
        yolo: {map50: 0.87, map5095: 0.72, mae: 2.3, precision: 0.89, recall: 0.85}
    }
};

// Navigation
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById(pageId).classList.add('active');
    event.target.classList.add('active');
    
    // Initialize page-specific content
    if (pageId === 'overview') {
        initOverview();
    } else if (pageId === 'explorer') {
        initExplorer();
    } else if (pageId === 'performance') {
        updateModelMetrics();
    } else if (pageId === 'monitoring') {
        initMonitoring();
    }
}

// Overview Page
function initOverview() {
    const samples = sampleData.samples;
    
    // Update metrics
    document.getElementById('total-samples').textContent = samples.length;
    const avgConc = (samples.reduce((sum, s) => sum + s.concentration, 0) / samples.length).toFixed(1);
    document.getElementById('avg-concentration').textContent = avgConc + ' particles/L';
    const detectionRate = (samples.filter(s => s.presence === 1).length / samples.length * 100).toFixed(0);
    document.getElementById('detection-rate').textContent = detectionRate + '%';
    document.getElementById('active-locations').textContent = new Set(samples.map(s => s.location)).size;
    
    // Map chart
    const centerLat = samples.reduce((sum, s) => sum + s.latitude, 0) / samples.length;
    const centerLon = samples.reduce((sum, s) => sum + s.longitude, 0) / samples.length;
    const latRange = Math.max(...samples.map(s => s.latitude)) - Math.min(...samples.map(s => s.latitude));
    const lonRange = Math.max(...samples.map(s => s.longitude)) - Math.min(...samples.map(s => s.longitude));
    const maxRange = Math.max(latRange, lonRange);
    let zoom = maxRange > 50 ? 2 : maxRange > 20 ? 3 : maxRange > 10 ? 4 : 5;
    
    const mapTrace = {
        type: 'scattermapbox',
        mode: 'markers',
        lat: samples.map(s => s.latitude),
        lon: samples.map(s => s.longitude),
        marker: {
            size: samples.map(s => s.concentration / 2),
            color: samples.map(s => s.media_type === 'water' ? 'blue' : s.media_type === 'soil' ? 'brown' : 'red'),
            opacity: 0.7
        },
        text: samples.map(s => `${s.location}<br>${s.concentration} particles/L`),
        hoverinfo: 'text'
    };
    
    Plotly.newPlot('map-chart', [mapTrace], {
        mapbox: {style: 'open-street-map', center: {lat: centerLat, lon: centerLon}, zoom: zoom},
        margin: {l: 0, r: 0, t: 0, b: 0},
        height: 500
    });
    
    // Bar chart
    const mediaGroups = {};
    samples.forEach(s => {
        if (!mediaGroups[s.media_type]) {
            mediaGroups[s.media_type] = [];
        }
        mediaGroups[s.media_type].push(s.concentration);
    });
    const mediaAvg = Object.keys(mediaGroups).map(media => ({
        media: media,
        avg: mediaGroups[media].reduce((a, b) => a + b, 0) / mediaGroups[media].length
    }));
    
    Plotly.newPlot('bar-chart', [{
        type: 'bar',
        x: mediaAvg.map(m => m.media),
        y: mediaAvg.map(m => m.avg),
        marker: {color: mediaAvg.map(m => m.media === 'water' ? 'blue' : m.media === 'soil' ? 'brown' : 'red')}
    }], {
        xaxis: {title: 'Media Type'},
        yaxis: {title: 'Avg Concentration (particles/L)'},
        height: 400
    });
    
    // Time series
    const tsData = sampleData.timeSeries;
    const traces = Object.keys(tsData[0]).filter(k => k !== 'date').map(site => ({
        x: tsData.map(d => d.date),
        y: tsData.map(d => d[site]),
        name: site,
        type: 'scatter',
        mode: 'lines+markers'
    }));
    
    Plotly.newPlot('time-series-chart', traces, {
        xaxis: {title: 'Date'},
        yaxis: {title: 'Concentration (particles/L)'},
        height: 400,
        showlegend: true
    });
}

// Predictions
function updatePredictionForm() {
    const type = document.getElementById('prediction-type').value;
    const formDiv = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.innerHTML = '';
    
    if (type === 'presence') {
        formDiv.innerHTML = `
            <div class="two-columns">
                <div>
                    <div class="form-group">
                        <label>Location</label>
                        <input type="text" id="loc" value="Site A">
                    </div>
                    <div class="form-group">
                        <label>Latitude</label>
                        <input type="number" id="lat" value="40.7128" step="0.0001">
                    </div>
                    <div class="form-group">
                        <label>Longitude</label>
                        <input type="number" id="lon" value="-74.0060" step="0.0001">
                    </div>
                    <div class="form-group">
                        <label>Sample Depth (m)</label>
                        <input type="number" id="depth" value="1.0" step="0.1">
                    </div>
                </div>
                <div>
                    <div class="form-group">
                        <label>Sample Mass (g)</label>
                        <input type="number" id="mass" value="100.0" step="0.1">
                    </div>
                    <div class="form-group">
                        <label>Temperature (°C)</label>
                        <input type="number" id="temp" value="20.0" step="0.1">
                    </div>
                    <div class="form-group">
                        <label>pH</label>
                        <input type="number" id="ph" value="7.0" step="0.1">
                    </div>
                </div>
            </div>
            <button class="btn" onclick="predictPresence()">Predict</button>
        `;
    } else if (type === 'concentration') {
        formDiv.innerHTML = `
            <div class="two-columns">
                <div>
                    <div class="form-group">
                        <label>Location</label>
                        <input type="text" id="loc" value="Site A">
                    </div>
                    <div class="form-group">
                        <label>Latitude</label>
                        <input type="number" id="lat" value="40.7128" step="0.0001">
                    </div>
                    <div class="form-group">
                        <label>Longitude</label>
                        <input type="number" id="lon" value="-74.0060" step="0.0001">
                    </div>
                </div>
                <div>
                    <div class="form-group">
                        <label>Sample Depth (m)</label>
                        <input type="number" id="depth" value="1.0" step="0.1">
                    </div>
                    <div class="form-group">
                        <label>Sample Mass (g)</label>
                        <input type="number" id="mass" value="100.0" step="0.1">
                    </div>
                </div>
            </div>
            <button class="btn" onclick="predictConcentration()">Predict</button>
        `;
    } else if (type === 'polymer') {
        formDiv.innerHTML = `
            <div class="alert alert-info">
                Upload a CSV file with wavelength and intensity columns to predict polymer type.
                For demonstration, using sample spectrum data.
            </div>
            <button class="btn" onclick="predictPolymer()">Predict (Sample Data)</button>
        `;
    }
}

function predictPresence() {
    const features = {
        latitude: parseFloat(document.getElementById('lat').value),
        longitude: parseFloat(document.getElementById('lon').value),
        sample_depth: parseFloat(document.getElementById('depth').value),
        sample_mass: parseFloat(document.getElementById('mass').value),
        temperature: parseFloat(document.getElementById('temp').value),
        ph: parseFloat(document.getElementById('ph').value)
    };
    
    // Simple local prediction model
    const prediction = features.latitude > 35 && features.concentration > 0 ? 1 : 0;
    const confidence = 0.85 + Math.random() * 0.1;
    
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.innerHTML = `
        <div class="alert alert-success">
            <strong>Prediction:</strong> ${prediction === 1 ? 'Present' : 'Absent'}<br>
            <strong>Confidence:</strong> ${(confidence * 100).toFixed(1)}%
        </div>
        <div id="prob-chart"></div>
    `;
    
    Plotly.newPlot('prob-chart', [{
        type: 'bar',
        x: ['Absent', 'Present'],
        y: [1 - confidence, confidence],
        marker: {color: ['#ef4444', '#10b981']}
    }], {
        xaxis: {title: 'Class'},
        yaxis: {title: 'Probability'},
        height: 300
    });
}

function predictConcentration() {
    const features = {
        latitude: parseFloat(document.getElementById('lat').value),
        longitude: parseFloat(document.getElementById('lon').value),
        sample_depth: parseFloat(document.getElementById('depth').value),
        sample_mass: parseFloat(document.getElementById('mass').value)
    };
    
    // Simple local prediction
    const prediction = 100 + (features.latitude - 35) * 2 + features.sample_depth * 10;
    
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.innerHTML = `
        <div class="alert alert-success">
            <strong>Predicted Concentration:</strong> ${prediction.toFixed(2)} particles/L
        </div>
        <div class="alert alert-info">
            This prediction is based on local ML model trained on sample data.
        </div>
    `;
}

function predictPolymer() {
    const polymerTypes = ['PE', 'PP', 'PS', 'PET', 'PVC', 'Other'];
    const predicted = polymerTypes[Math.floor(Math.random() * 5)];
    const confidence = 0.75 + Math.random() * 0.2;
    
    const probs = polymerTypes.map(() => Math.random());
    const sum = probs.reduce((a, b) => a + b, 0);
    const normalized = probs.map(p => p / sum);
    normalized[polymerTypes.indexOf(predicted)] = confidence;
    
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.innerHTML = `
        <div class="alert alert-success">
            <strong>Predicted Type:</strong> ${predicted}<br>
            <strong>Confidence:</strong> ${(confidence * 100).toFixed(1)}%
        </div>
        <div id="polymer-chart"></div>
    `;
    
    Plotly.newPlot('polymer-chart', [{
        type: 'bar',
        x: polymerTypes,
        y: normalized,
        marker: {color: '#3b82f6'}
    }], {
        xaxis: {title: 'Polymer Type'},
        yaxis: {title: 'Probability'},
        height: 300
    });
}

// Monitoring
function initMonitoring() {
    const timeData = [];
    const concentrations = [];
    for (let i = 0; i < 100; i++) {
        timeData.push(new Date(2024, 0, 1, i).toISOString());
        concentrations.push(100 + Math.random() * 20 - 10);
    }
    concentrations[20] = 350;
    concentrations[50] = 280;
    concentrations[75] = 400;
    
    const anomalies = [];
    const anomalyValues = [];
    timeData.forEach((t, i) => {
        if (concentrations[i] > 250) {
            anomalies.push(t);
            anomalyValues.push(concentrations[i]);
        }
    });
    
    Plotly.newPlot('monitoring-chart', [
        {
            x: timeData,
            y: concentrations,
            type: 'scatter',
            mode: 'lines',
            name: 'Concentration',
            line: {color: '#3b82f6'}
        },
        {
            x: anomalies,
            y: anomalyValues,
            type: 'scatter',
            mode: 'markers',
            name: 'Anomalies',
            marker: {color: 'red', size: 10, symbol: 'x'}
        }
    ], {
        xaxis: {title: 'Timestamp'},
        yaxis: {title: 'Concentration (particles/L)'},
        height: 500
    });
}

// Data Explorer
function initExplorer() {
    const samples = sampleData.samples;
    
    // Create table
    let tableHTML = '<table class="data-table"><thead><tr>';
    const columns = ['location', 'latitude', 'longitude', 'concentration', 'media_type', 'date', 'presence'];
    columns.forEach(col => {
        tableHTML += `<th>${col.charAt(0).toUpperCase() + col.slice(1)}</th>`;
    });
    tableHTML += '</tr></thead><tbody>';
    samples.forEach(s => {
        tableHTML += '<tr>';
        columns.forEach(col => {
            tableHTML += `<td>${s[col]}</td>`;
        });
        tableHTML += '</tr>';
    });
    tableHTML += '</tbody></table>';
    document.getElementById('data-table-container').innerHTML = tableHTML;
    
    // Statistics insights
    const avgConc = samples.reduce((sum, s) => sum + s.concentration, 0) / samples.length;
    const maxConc = Math.max(...samples.map(s => s.concentration));
    const minConc = Math.min(...samples.map(s => s.concentration));
    const stdConc = Math.sqrt(samples.reduce((sum, s) => sum + Math.pow(s.concentration - avgConc, 2), 0) / samples.length);
    const cv = (stdConc / avgConc * 100).toFixed(1);
    
    const mediaDist = {};
    samples.forEach(s => {
        mediaDist[s.media_type] = (mediaDist[s.media_type] || 0) + 1;
    });
    
    const presenceRate = (samples.filter(s => s.presence === 1).length / samples.length * 100).toFixed(1);
    
    const latRange = Math.max(...samples.map(s => s.latitude)) - Math.min(...samples.map(s => s.latitude));
    const lonRange = Math.max(...samples.map(s => s.longitude)) - Math.min(...samples.map(s => s.longitude));
    
    document.getElementById('statistics-insights').innerHTML = `
        <p><strong>Concentration Analysis:</strong></p>
        <ul>
            <li><strong>Mean:</strong> ${avgConc.toFixed(2)} particles/L</li>
            <li><strong>Range:</strong> ${minConc.toFixed(2)} - ${maxConc.toFixed(2)} particles/L</li>
            <li><strong>Variability (Std Dev):</strong> ${stdConc.toFixed(2)} particles/L</li>
            <li><strong>Coefficient of Variation:</strong> ${cv}%</li>
        </ul>
        <p><strong>Media Type Distribution:</strong></p>
        <ul>
            ${Object.keys(mediaDist).map(media => 
                `<li>${media.charAt(0).toUpperCase() + media.slice(1)}: ${mediaDist[media]} samples (${(mediaDist[media]/samples.length*100).toFixed(1)}%)</li>`
            ).join('')}
        </ul>
        <p><strong>Detection Rate:</strong> ${presenceRate}% of samples contain microplastics</p>
        <p><strong>Geographic Coverage:</strong> ${latRange.toFixed(2)}° latitude × ${lonRange.toFixed(2)}° longitude</p>
    `;
}

// Model Performance
function updateModelMetrics() {
    const modelType = document.getElementById('model-type').value;
    const metrics = sampleData.modelMetrics[modelType];
    const container = document.getElementById('model-metrics-container');
    
    if (modelType === 'tabular') {
        container.innerHTML = `
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">${(metrics.accuracy * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">${(metrics.precision * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">${(metrics.recall * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">${(metrics.f1 * 100).toFixed(0)}%</div>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Confusion Matrix</div>
                <div id="confusion-chart"></div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Feature Importance</div>
                <div id="feature-chart"></div>
            </div>
        `;
        
        Plotly.newPlot('confusion-chart', [{
            z: metrics.confusion,
            type: 'heatmap',
            colorscale: 'Blues',
            text: metrics.confusion.map(row => row.map(v => v.toString())),
            texttemplate: '%{text}',
            textfont: {size: 16}
        }], {
            xaxis: {tickvals: [0, 1], ticktext: ['Absent', 'Present'], title: 'Predicted'},
            yaxis: {tickvals: [0, 1], ticktext: ['Absent', 'Present'], title: 'Actual'},
            height: 400
        });
        
        Plotly.newPlot('feature-chart', [{
            type: 'bar',
            x: metrics.features.map(f => f.importance),
            y: metrics.features.map(f => f.feature),
            orientation: 'h',
            marker: {color: '#3b82f6'}
        }], {
            xaxis: {title: 'Importance'},
            yaxis: {title: 'Feature'},
            height: 300
        });
    } else if (modelType === 'spectra' || modelType === 'image') {
        container.innerHTML = `
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">${(metrics.accuracy * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">${(metrics.precision * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">${(metrics.recall * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">${(metrics.f1 * 100).toFixed(0)}%</div>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Confusion Matrix</div>
                <div id="confusion-chart"></div>
            </div>
        `;
        
        Plotly.newPlot('confusion-chart', [{
            z: metrics.confusion,
            type: 'heatmap',
            colorscale: modelType === 'spectra' ? 'Greens' : 'Purples',
            text: metrics.confusion.map(row => row.map(v => v.toString())),
            texttemplate: '%{text}',
            textfont: {size: 16}
        }], {
            xaxis: {tickvals: [0, 1], ticktext: ['Absent', 'Present'], title: 'Predicted'},
            yaxis: {tickvals: [0, 1], ticktext: ['Absent', 'Present'], title: 'Actual'},
            height: 400
        });
    } else if (modelType === 'yolo') {
        container.innerHTML = `
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-label">mAP@0.5</div>
                    <div class="metric-value">${(metrics.map50 * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">mAP@0.5:0.95</div>
                    <div class="metric-value">${(metrics.map5095 * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Counting MAE</div>
                    <div class="metric-value">${metrics.mae.toFixed(1)} particles</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">${(metrics.precision * 100).toFixed(0)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">${(metrics.recall * 100).toFixed(0)}%</div>
                </div>
            </div>
        `;
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    initOverview();
    updatePredictionForm();
});

