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
    
    // We have the query, now we need to generate a response, after loading we will call the API
    setTimeout(() => {
        generateResponse(query).then(response => {
            showResults(response, query);
        }).catch(error => {
            showError('Failed to fetch results from the backend.');
            console.error('Error fetching response:', error);
        });
    }, 1500);
}

// Call the API to generate a response
async function generateResponse(query) {
    const response = await fetch('api/query', { // this is vercel expectation, if localhost use http://localhost:8000/query
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
    });
    const data = await response.json();
    return data.response;
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
        <div class="response-display">
            ${response}
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

