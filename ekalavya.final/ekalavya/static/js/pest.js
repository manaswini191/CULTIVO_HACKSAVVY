let sections = document.querySelectorAll('section');
let navLinks = document.querySelectorAll('header nav a');

window.onscroll = () => {
    sections.forEach(sec => {
        let top = window.scrollY;
        let offset = sec.offsetTop;
        let height = sec.offsetHeight;
        let id = sec.getAttribute('id');
        
        if(top >= offset && top < offset + height){
            navLinks.forEach(links => {
                links.classList.remove('active');
                document.querySelector('header nav a[href*=' +id +']').classList.add('active');
            });
        };
    });
};
let profile = document.querySelector(".profile");
let subMenu = document.getElementById("subMenu");
profile.addEventListener("click", function() {
    subMenu.classList.toggle("open-menu");
});
const selectImage = document.querySelector('.select-image');
const inputFile = document.querySelector('#file');
const imgArea = document.querySelector('.img-area');

selectImage.addEventListener('click', function () {
    inputFile.click();
})
inputFile.addEventListener('change', function () {
    const image = this.files[0]
    if (image.size < 2000000) {
        const reader = new FileReader();
        reader.onload = () => {
            const allImg = imgArea.querySelectorAll('img');
            allImg.forEach(item => item.remove());
            const imgUrl = reader.result;
            const img = document.createElement('img');
            img.src = imgUrl;
            imgArea.appendChild(img);
            imgArea.classList.add('active');
            imgArea.dataset.img = image.name;
        }
        reader.readAsDataURL(image);
    } else {
        alert("Image size more than 2MB");
    }
});
// Sample crop names
const cropNames =  ["Wheat",
"Rice",
"Maize",
"Barley",
"Oats",
"Sorghum",
"Millet",
"Quinoa",
"Sunflower",
"Cotton",
"Soybeans",
"Peanuts",
"Potatoes",
"Tomatoes",
"Cabbage",
"Carrots",
"Lettuce",
"Spinach",
"Broccoli",
"Cucumbers",
"Peppers",
"Onions",
"Garlic",
"Strawberries",
"Apples",
"Grapes",
"Oranges",
"Bananas"]
// Get the dropdown element
const dropdown = document.getElementById('crop-dropdown');

// Populate the dropdown with crop names
cropNames.forEach(cropName => {
    const option = document.createElement('option');
    option.text = cropName;
    dropdown.add(option);
});
