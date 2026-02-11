// Variables globales
let currentMode = 'camera';

// Initialisation
document.addEventListener('DOMContentLoaded', function() {
    // Charger la liste des modèles
    loadModelsList();
    
    // Mettre à jour les stats de la caméra
    updateCameraStats();
    setInterval(updateCameraStats, 1000);
    
    // Gestion drag and drop
    setupDragAndDrop();
});

// Charger la liste des modèles
function loadModelsList() {
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const select = document.getElementById('modelSelect');
                select.innerHTML = '';
                
                if (data.models.length === 0) {
                    select.innerHTML = '<option value="">Aucun modèle disponible</option>';
                    return;
                }
                
                // Ajouter une option par défaut
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = '-- Sélectionnez un modèle --';
                select.appendChild(defaultOption);
                
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.filename;
                    option.textContent = model.display_name;
                    select.appendChild(option);
                });
                
                // Charger les infos du modèle actuel
                updateModelInfo();
            }
        })
        .catch(error => console.error('Erreur chargement modèles:', error));
}

// Charger le modèle sélectionné
function loadSelectedModel() {
    const select = document.getElementById('modelSelect');
    const modelName = select.value;
    
    if (!modelName) {
        showModelStatus('Veuillez sélectionner un modèle', 'error');
        return;
    }
    
    fetch('/api/load_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showModelStatus('Modèle chargé avec succès', 'success');
            updateModelInfo();
            
            // Ne pas recharger la page - simplement mettre à jour le flux vidéo si actif
            if (currentMode === 'camera') {
                refreshVideo();
            }
        } else {
            showModelStatus('Erreur: ' + data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Erreur:', error);
        showModelStatus('Erreur lors du chargement', 'error');
    });
}

// Rafraîchir le flux vidéo sans changer de mode
function refreshVideo() {
    const videoFeed = document.getElementById('videoFeed');
    // Forcer le rafraîchissement du flux vidéo
    videoFeed.src = '/video_feed?' + new Date().getTime();
    
    // Réinitialiser les stats locales
    fetch('/api/stats/reset', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateCameraStats();
        }
    });
}

// Afficher un message de statut discret sous le sélecteur de modèle
function showModelStatus(message, type) {
    const statusElement = document.getElementById('modelStatusMessage');
    
    // Définir le message et le style
    statusElement.textContent = message;
    statusElement.className = 'model-status-message';
    
    if (type === 'success') {
        statusElement.classList.add('model-status-success');
    } else if (type === 'error') {
        statusElement.classList.add('model-status-error');
    }
    
    statusElement.style.display = 'block';
    
    // Masquer le message après 3 secondes
    setTimeout(() => {
        statusElement.style.display = 'none';
    }, 3000);
}

// Mettre à jour les infos du modèle
function updateModelInfo() {
    fetch('/api/model_info')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('currentModelName').textContent = data.model_name || 'Aucun';
                document.getElementById('currentModelType').textContent = data.model_type || '-';
                document.getElementById('currentModelArchitecture').textContent = data.architecture || '-';
            }
        });
}

// Changer de mode (caméra/image)
function switchMode(mode) {
    currentMode = mode;
    
    // Mettre à jour les boutons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.querySelector('.status-indicator').className = 'status-indicator status-inactive';
    });
    
    const activeBtn = document.querySelector(`.mode-btn[onclick="switchMode('${mode}')"]`);
    activeBtn.classList.add('active');
    activeBtn.querySelector('.status-indicator').className = 'status-indicator status-active';
    
    // Afficher/masquer les sections
    document.getElementById('camera-section').classList.remove('active');
    document.getElementById('image-section').classList.remove('active');
    
    if (mode === 'camera') {
        document.getElementById('camera-section').classList.add('active');
        // Rafraîchir le flux vidéo
        refreshVideo();
    } else {
        document.getElementById('image-section').classList.add('active');
    }
}

// Mettre à jour les stats de la caméra
function updateCameraStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('cameraFrames').textContent = data.stats.frames_processed;
                document.getElementById('cameraFaces').textContent = data.stats.faces_detected;
                document.getElementById('cameraRate').textContent = data.stats.detection_rate + '%';
                document.getElementById('cameraYoung').textContent = data.stats.young_count;
                document.getElementById('cameraAdult').textContent = data.stats.adult_count;
                document.getElementById('cameraSenior').textContent = data.stats.senior_count;
            }
        })
        .catch(error => console.error('Erreur stats:', error));
}

// Gestion de l'upload d'image
document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.match('image.*')) {
        alert('Veuillez sélectionner une image valide');
        return;
    }
    
    const preview = document.getElementById('imagePreview');
    const reader = new FileReader();
    
    reader.onload = function(e) {
        preview.innerHTML = `
            <div style="text-align: center;">
                <img src="${e.target.result}" alt="Aperçu">
                <div class="loader" id="imageLoader"></div>
            </div>
        `;
        
        const formData = new FormData();
        formData.append('image', file);
        
        document.getElementById('textResults').style.display = 'none';
        
        fetch('/api/analyze_upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Afficher l'image originale
                preview.innerHTML = `
                    <div style="text-align: center;">
                        <img src="${data.image_original}" alt="Image analysée">
                        <div style="margin-top: 10px; color: #666; font-size: 0.9em;">
                            ${data.num_faces} visage(s) détecté(s)
                        </div>
                    </div>
                `;
                
                // Afficher les résultats texte (plage d'âge seulement)
                const textResults = document.getElementById('textResults');
                const detectionCount = document.getElementById('detectionCount');
                const predictionsList = document.getElementById('predictionsList');
                
                detectionCount.textContent = data.message;
                predictionsList.innerHTML = '';
                
                // Afficher chaque prédiction (plage d'âge seulement)
                data.predictions.forEach((pred, index) => {
                    const p = pred.prediction;
                    const predDiv = document.createElement('div');
                    predDiv.className = `prediction-item ${p.interface_class}`;
                    
                    // Déterminer l'icône et le label
                    let icon, label;
                    if (p.interface_class === 'young') {
                        icon = '👶';
                        label = 'Jeune';
                    } else if (p.interface_class === 'adult') {
                        icon = '🧑';
                        label = 'Adulte';
                    } else {
                        icon = '👵';
                        label = 'Senior';
                    }
                    
                    predDiv.innerHTML = `
                        <div class="prediction-info">
                            <div class="prediction-class">
                                ${icon} ${label}
                            </div>
                            <div class="prediction-range">
                                Plage d'âge: ${p.age_range}
                            </div>
                        </div>
                        <div class="prediction-confidence">
                            ${p.confidence_percent}%
                        </div>
                    `;
                    
                    predictionsList.appendChild(predDiv);
                });
                
                textResults.style.display = 'block';
                
            } else {
                preview.innerHTML = `
                    <div style="color: #f56565; padding: 20px; text-align: center;">
                        ❌ Erreur: ${data.message}
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            preview.innerHTML = `
                <div style="color: #f56565; padding: 20px; text-align: center;">
                    ❌ Erreur lors de l'analyse
                </div>
            `;
        });
    };
    
    reader.readAsDataURL(file);
});

// Configuration drag and drop
function setupDragAndDrop() {
    const uploadArea = document.querySelector('.upload-area');
    
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.backgroundColor = '#f0f7ff';
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#a0c4ff';
        uploadArea.style.backgroundColor = '#f8faff';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#a0c4ff';
        uploadArea.style.backgroundColor = '#f8faff';
        
        if (e.dataTransfer.files.length) {
            const file = e.dataTransfer.files[0];
            const fileInput = document.getElementById('fileInput');
            
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    });
}

// Contrôles de la caméra
function startCamera() {
    alert('Caméra démarrée ! Le flux vidéo est actif.');
}

function stopCamera() {
    if (confirm('Arrêter la détection en temps réel ?')) {
        location.reload();
    }
}

function resetStats() {
    if (confirm('Réinitialiser toutes les statistiques ?')) {
        fetch('/api/stats/reset', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateCameraStats();
                alert('Statistiques réinitialisées !');
            }
        });
    }
}

function exportStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const statsStr = JSON.stringify(data.stats, null, 2);
                const blob = new Blob([statsStr], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `statistiques_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                alert('Statistiques exportées avec succès !');
            }
        });
}