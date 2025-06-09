document.addEventListener('DOMContentLoaded', function() {
    const gallerySelect = document.getElementById('gall-select');
    const galleryImg = document.getElementById('gall-img');

    // Gallery data
    const galleryData = {
        'scene1': {
            title: 'Office Environment',
            images: [
                { src: 'assets/images/scene1/input.jpg', caption: 'Input Image' },
                { src: 'assets/images/scene1/semantic.jpg', caption: 'Semantic Segmentation' },
                { src: 'assets/images/scene1/localization.jpg', caption: 'Localization Result' }
            ]
        },
        'scene2': {
            title: 'Residential Space',
            images: [
                { src: 'assets/images/scene2/input.jpg', caption: 'Input Image' },
                { src: 'assets/images/scene2/semantic.jpg', caption: 'Semantic Segmentation' },
                { src: 'assets/images/scene2/localization.jpg', caption: 'Localization Result' }
            ]
        },
        'scene3': {
            title: 'Commercial Building',
            images: [
                { src: 'assets/images/scene3/input.jpg', caption: 'Input Image' },
                { src: 'assets/images/scene3/semantic.jpg', caption: 'Semantic Segmentation' },
                { src: 'assets/images/scene3/localization.jpg', caption: 'Localization Result' }
            ]
        }
    };

    // Handle gallery selection
    gallerySelect.addEventListener('change', function() {
        const selectedScene = this.value;
        
        if (selectedScene === 'none') {
            galleryImg.innerHTML = '';
            return;
        }

        const scene = galleryData[selectedScene];
        let galleryHTML = `<h5 class="mb-4">${scene.title}</h5>`;
        galleryHTML += '<div class="row">';

        scene.images.forEach(image => {
            galleryHTML += `
                <div class="col-md-4 mb-4">
                    <div class="card">
                        <img src="${image.src}" class="card-img-top" alt="${image.caption}">
                        <div class="card-body">
                            <p class="card-text text-center">${image.caption}</p>
                        </div>
                    </div>
                </div>
            `;
        });

        galleryHTML += '</div>';
        galleryImg.innerHTML = galleryHTML;
    });
}); 