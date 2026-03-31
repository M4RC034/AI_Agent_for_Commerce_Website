document.addEventListener('DOMContentLoaded', async () => {
    const loadingIndicator = document.getElementById('loadingIndicator');
    const ctx = document.getElementById('categoryChart').getContext('2d');
    
    // Smooth, modern color palette for the dark glassmorphism theme
    const colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
        '#FFEEAD', '#D4A5A5', '#9B59B6', '#3498DB', 
        '#E67E22', '#1ABC9C', '#F1C40F', '#E74C3C', 
        '#2ECC71', '#34495E', '#95A5A6'
    ];

    try {
        const response = await fetch('http://127.0.0.1:8000/api/statistics');
        if (!response.ok) throw new Error('Failed to fetch statistics');
        
        const data = await response.json();
        const categories = data.categories;
        
        // Hide loader
        loadingIndicator.style.display = 'none';

        // Prepare data arrays
        const labels = Object.keys(categories);
        const counts = Object.values(categories);
        
        // Calculate total for percentage math
        const totalProducts = counts.reduce((acc, current) => acc + current, 0);

        // Render Chart.js Pie Chart
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: colors.slice(0, labels.length),
                    borderColor: 'rgba(25, 25, 25, 1)',
                    borderWidth: 2,
                    hoverOffset: 12
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.8)',
                            padding: 20,
                            font: {
                                family: "'Plus Jakarta Sans', sans-serif",
                                size: 13
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleFont: {
                            family: "'Plus Jakarta Sans', sans-serif",
                            size: 14,
                            weight: 'bold'
                        },
                        bodyFont: {
                            family: "'Plus Jakarta Sans', sans-serif",
                            size: 14
                        },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            // Format hover tooltip to show exact percentage calculation
                            label: function(context) {
                                let label = context.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                const value = context.parsed;
                                const percentage = ((value / totalProducts) * 100).toFixed(2);
                                label += `${value} items (${percentage}%)`;
                                return label;
                            }
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error('Error loading stats:', error);
        loadingIndicator.innerHTML = '<p style="color: #FF6B6B">Failed to load statistics. Is the backend running?</p>';
    }
});
