const serverUrl = 'http://127.0.0.1:65501';

document.getElementById('input-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const username = document.getElementById('username').value;
    const category = document.getElementById('category').value;

    const response = await fetch(`${serverUrl}/agents`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, category })
    });

    const data = await response.json();

    displayResults(data);
});
function displayResults(data) {
    const thumbnailsContainer = document.getElementById('thumbnails-container');
    thumbnailsContainer.innerHTML = '';
    data.imageUrls.forEach(url => {
        const img = document.createElement('img');
        img.src = url;
        thumbnailsContainer.appendChild(img);
    });

    document.getElementById('title-container').innerText = data.title;
    document.getElementById('description-container').innerText = data.description;
    document.getElementById('script-container').innerText = data.script;
}