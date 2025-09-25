// DOM elements
const queryInput = document.getElementById('queryInput');
const searchBtn = document.getElementById('searchBtn');
const resultsSection = document.getElementById('resultsSection');
const resultContent = document.getElementById('resultContent');
const loading = document.getElementById('loading');

// Simple demo response
const demoResponse = "I found several apartment options that match your criteria. The listings include various sizes, price points, and amenities. Would you like me to provide more specific details about any particular aspect, such as location, price range, or features?";

// Event listeners
searchBtn.addEventListener('click', handleSearch);
queryInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        handleSearch();
    }
});

// Handle search functionality
function handleSearch() {
    const query = queryInput.value.trim();
    
    if (!query) {
        showError('Please enter a question about apartments.');
        return;
    }
    
    // Show loading state
    showLoading();
    
    // Simulate API call delay
    setTimeout(() => {
        const response = generateResponse(query);
        showResults(response, query);
    }, 1500);
}

// Generate a response based on the query
function generateResponse(query) {
    return demoResponse;
}

// Show loading state
function showLoading() {
    resultsSection.style.display = 'none';
    loading.style.display = 'block';
    searchBtn.disabled = true;
    searchBtn.innerHTML = '<span class="btn-text">Searching...</span><span class="btn-icon">⏳</span>';
}

// Show results
function showResults(response, query) {
    loading.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Reset button
    searchBtn.disabled = false;
    searchBtn.innerHTML = '<span class="btn-text">Search</span><span class="btn-icon">🔍</span>';
    
    // Display results
    resultContent.innerHTML = `
        <div class="query-display">
            <strong>Your question:</strong> "${query}"
        </div>
        <div class="response-display">
            <strong>Answer:</strong><br>
            ${response}
        </div>
        <div class="demo-notice">
            <em>Note: This is a demo interface. In the full version, this would connect to your RAG pipeline to provide real answers from the apartment listings.</em>
        </div>
    `;
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Show error message
function showError(message) {
    resultsSection.style.display = 'block';
    loading.style.display = 'none';
    
    resultContent.innerHTML = `
        <div class="error-message">
            <strong>⚠️ ${message}</strong>
        </div>
    `;
    
    // Reset button
    searchBtn.disabled = false;
    searchBtn.innerHTML = '<span class="btn-text">Search</span><span class="btn-icon">🔍</span>';
}

// Add some interactive features
queryInput.addEventListener('input', function() {
    // Clear results when user starts typing new query
    if (resultsSection.style.display !== 'none') {
        resultsSection.style.display = 'none';
    }
});

// Add focus effect
queryInput.addEventListener('focus', function() {
    this.parentElement.style.transform = 'scale(1.02)';
});

queryInput.addEventListener('blur', function() {
    this.parentElement.style.transform = 'scale(1)';
});

