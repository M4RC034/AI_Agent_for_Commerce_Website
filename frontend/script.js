document.addEventListener('DOMContentLoaded', () => {
    // Time-aware greeting logic
    const greetingLogo = document.getElementById('greetingLogo');
    if (greetingLogo) {
        const hour = new Date().getHours();
        if (hour >= 5 && hour < 12) greetingLogo.textContent = 'Good morning';
        else if (hour >= 12 && hour < 17) greetingLogo.textContent = 'Good afternoon';
        else if (hour >= 17 && hour < 22) greetingLogo.textContent = 'Good evening';
        else greetingLogo.textContent = 'Good night';
    }

    const form = document.getElementById('searchForm');
    const imageUpload = document.getElementById('imageUpload');
    const textInput = document.getElementById('textInput');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const imageName = document.getElementById('imageName');
    const removeImageBtn = document.getElementById('removeImageBtn');

    let currentFile = null;

    // ── Multi-Turn Conversation History ──
    let conversationHistory = [];

    // ── Search History (persisted in localStorage) ──
    const HISTORY_KEY = 'al_search_history';
    const MAX_HISTORY = 20;
    const infoToggleBtn = document.getElementById('infoToggleBtn');
    const infoPanel = document.getElementById('infoPanel');
    const historyToggleBtn = document.getElementById('historyToggleBtn');
    const historyPanel = document.getElementById('historyPanel');
    const historyList = document.getElementById('historyList');
    const historyEmpty = document.getElementById('historyEmpty');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');

    function loadSearchHistory() {
        try {
            return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
        } catch { return []; }
    }

    function saveSearchHistory(history) {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    }

    function addToSearchHistory(query) {
        if (!query || query === 'Please find items visually similar to this image.') return;
        let history = loadSearchHistory();
        // Remove duplicate if it exists
        history = history.filter(h => h.query !== query);
        // Add to the front
        history.unshift({ query, timestamp: Date.now() });
        // Cap at MAX_HISTORY
        if (history.length > MAX_HISTORY) history = history.slice(0, MAX_HISTORY);
        saveSearchHistory(history);
    }

    function formatTimeAgo(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);
        if (seconds < 60) return 'just now';
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    }

    function renderSearchHistory() {
        const history = loadSearchHistory();
        historyList.innerHTML = '';

        if (history.length === 0) {
            historyEmpty.style.display = 'block';
            return;
        }

        historyEmpty.style.display = 'none';
        history.forEach(item => {
            const li = document.createElement('li');
            li.className = 'history-item';
            li.innerHTML = `
                <i data-lucide="search" class="history-icon" style="width:14px;height:14px;"></i>
                <span class="history-text">${item.query}</span>
                <span class="history-time">${formatTimeAgo(item.timestamp)}</span>
            `;
            li.addEventListener('click', () => {
                textInput.value = item.query;
                historyPanel.classList.add('hidden');
                historyToggleBtn.classList.remove('active');
                textInput.focus();
            });
            historyList.appendChild(li);
        });

        // Re-render lucide icons for the newly added search icons
        lucide.createIcons();
    }

    // Toggle info panel
    infoToggleBtn.addEventListener('click', () => {
        const isOpen = !infoPanel.classList.contains('hidden');
        if (isOpen) {
            infoPanel.classList.add('hidden');
            infoToggleBtn.classList.remove('active');
        } else {
            // Close history panel if open
            historyPanel.classList.add('hidden');
            historyToggleBtn.classList.remove('active');
            
            infoPanel.classList.remove('hidden');
            infoToggleBtn.classList.add('active');
        }
    });

    // Toggle history panel
    historyToggleBtn.addEventListener('click', () => {
        const isOpen = !historyPanel.classList.contains('hidden');
        if (isOpen) {
            historyPanel.classList.add('hidden');
            historyToggleBtn.classList.remove('active');
        } else {
            // Close info panel if open
            infoPanel.classList.add('hidden');
            infoToggleBtn.classList.remove('active');

            renderSearchHistory();
            historyPanel.classList.remove('hidden');
            historyToggleBtn.classList.add('active');
        }
    });

    // Close panels when clicking outside
    document.addEventListener('click', (e) => {
        if (!historyPanel.classList.contains('hidden') &&
            !historyPanel.contains(e.target) &&
            !historyToggleBtn.contains(e.target)) {
            historyPanel.classList.add('hidden');
            historyToggleBtn.classList.remove('active');
        }
        
        if (!infoPanel.classList.contains('hidden') &&
            !infoPanel.contains(e.target) &&
            !infoToggleBtn.contains(e.target)) {
            infoPanel.classList.add('hidden');
            infoToggleBtn.classList.remove('active');
        }
    });

    // Clear history
    clearHistoryBtn.addEventListener('click', () => {
        saveSearchHistory([]);
        renderSearchHistory();
    });

    // Handle Image Selection
    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            currentFile = file;
            const reader = new FileReader();

            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imageName.textContent = file.name;
                imagePreviewContainer.classList.remove('hidden');
                textInput.placeholder = "Add an optional message...";
            };

            reader.readAsDataURL(file);
        }
    });

    // Handle Remove Image
    removeImageBtn.addEventListener('click', () => {
        currentFile = null;
        imageUpload.value = '';
        imagePreviewContainer.classList.add('hidden');
        imagePreview.src = '';
        textInput.placeholder = "Ask Al anything, or upload a photo to find products...";
    });

    // Handle Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const message = textInput.value.trim();

        if (!message && !currentFile) {
            textInput.focus();
            return;
        }

        const resultsContainer = document.getElementById('resultsContainer');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const chatResponse = document.getElementById('chatResponse');
        const productGallery = document.getElementById('productGallery');

        // UI state for loading
        resultsContainer.classList.remove('hidden');
        loadingIndicator.classList.remove('hidden');
        chatResponse.innerHTML = '';
        productGallery.innerHTML = '';

        // Close panels if open
        historyPanel.classList.add('hidden');
        historyToggleBtn.classList.remove('active');
        infoPanel.classList.add('hidden');
        infoToggleBtn.classList.remove('active');

        // Prepare data
        const formData = new FormData();
        const sentMessage = message || "Please find items visually similar to this image.";
        formData.append('message', sentMessage);
        if (currentFile) {
            formData.append('image', currentFile);
        }
        formData.append('history_json', JSON.stringify(conversationHistory));

        // Save to search history
        if (message) addToSearchHistory(message);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error("Backend server responded with an error.");
            }

            const data = await response.json();

            loadingIndicator.classList.add('hidden');

            chatResponse.innerHTML = `<p>${data.agent_response.replace(/\n/g, '<br>')}</p>`;

            // Update multi-turn conversation history
            conversationHistory.push({ role: 'user', content: sentMessage });
            conversationHistory.push({ role: 'assistant', content: data.agent_response });
            if (conversationHistory.length > 10) {
                conversationHistory = conversationHistory.slice(-10);
            }

            // Handle Products Rendering
            if (data.products && data.products.length > 0) {
                const firstProd = data.products[0];
                if (firstProd.product_id === 'NONE') {
                    chatResponse.innerHTML += `<br><p class="error-msg">⚠️ ${firstProd.description}</p>`;
                } else {
                    data.products.forEach((prod) => {
                        const card = document.createElement('div');
                        card.className = 'product-card';

                        let priceDisp = prod.price;
                        if (!priceDisp.includes('N/A') && !priceDisp.startsWith('$')) {
                            priceDisp = `$${priceDisp}`;
                        }

                        const badgeHTML = prod.badge ? `<div class="xai-badge">${prod.badge}</div>` : '';

                        card.innerHTML = `
                            <div class="product-img-wrapper">
                                ${badgeHTML}
                                <img src="${prod.image_url}" alt="${prod.title}" onerror="this.src='https://via.placeholder.com/200x200?text=No+Photo';">
                            </div>
                            <div class="product-info">
                                <span class="product-category">${prod.category}</span>
                                <h3 class="product-title" title="${prod.title}">${prod.title}</h3>
                                <span class="product-price">${priceDisp}</span>
                                <a href="${prod.url}" target="_blank" class="buy-btn">View on Amazon</a>
                            </div>
                        `;
                        productGallery.appendChild(card);
                    });
                }
            }
        } catch (err) {
            loadingIndicator.classList.add('hidden');
            chatResponse.innerHTML = `<p class="error-msg">Failed to reach Al: ${err.message}. Ensure uvicorn backend is running.</p>`;
        }

        textInput.value = '';
    });
});
