// DOM elements
const queryInput = document.getElementById('queryInput');
const searchBtn = document.getElementById('searchBtn');
const resultsSection = document.getElementById('resultsSection');
const resultContent = document.getElementById('resultContent');
const loading = document.getElementById('loading');

// Sample responses for demo purposes
const sampleResponses = [
    "Based on the available listings, I found several apartments under $3000. The Berkeley Place has 1-bedroom units starting at $2,950/month, and there are options at 2067 University Ave with various sizes and amenities.",
    
    "I found multiple 2-bedroom apartments available. The Kittredge building has 2-bedroom, 2-bathroom units at $4,000/month, and there are several other options in the Berkeley area with different price points and features.",
    
    "For pet-friendly apartments, I can see several options that allow both cats and dogs. Most listings include pet policies with additional fees, typically around $150-200 per month for pets.",
    
    "The listings show various amenities including in-unit washer & dryer, stainless steel appliances, high-speed internet, and some buildings offer parking spaces. Many are located within walking distance of UC Berkeley campus."
];

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
    const lowerQuery = query.toLowerCase();
    
    // Simple keyword matching for demo
    if (lowerQuery.includes('3000') || lowerQuery.includes('under') || lowerQuery.includes('cheap')) {
        return sampleResponses[0];
    } else if (lowerQuery.includes('2') && (lowerQuery.includes('bedroom') || lowerQuery.includes('bed'))) {
        return sampleResponses[1];
    } else if (lowerQuery.includes('pet') || lowerQuery.includes('dog') || lowerQuery.includes('cat')) {
        return sampleResponses[2];
    } else if (lowerQuery.includes('amenit') || lowerQuery.includes('feature') || lowerQuery.includes('include')) {
        return sampleResponses[3];
    } else {
        // Default response
        return "I found several apartment options that match your criteria. The listings include various sizes, price points, and amenities. Would you like me to provide more specific details about any particular aspect, such as location, price range, or features?";
    }
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

// Add some example queries
const exampleQueries = [
    "What apartments are available for under $3000?",
    "Show me 2 bedroom apartments",
    "Are there any pet-friendly apartments?",
    "What amenities are included?"
];

// Add example queries to the page
document.addEventListener('DOMContentLoaded', function() {
    const examplesContainer = document.createElement('div');
    examplesContainer.className = 'examples-container';
    examplesContainer.innerHTML = `
        <h3>Try these example questions:</h3>
        <div class="example-queries">
            ${exampleQueries.map(query => 
                `<button class="example-btn" onclick="fillQuery('${query}')">${query}</button>`
            ).join('')}
        </div>
    `;
    
    const searchSection = document.querySelector('.search-section');
    searchSection.appendChild(examplesContainer);
});

// Function to fill query from examples
function fillQuery(query) {
    queryInput.value = query;
    queryInput.focus();
}
