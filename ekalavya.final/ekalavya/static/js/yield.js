const button = document.getElementById('get-location-button');
const temperatureDisplay = document.getElementById('temperature');
const humidityDisplay = document.getElementById('humidity');
const weatherConditionDisplay = document.getElementById('weather-condition');
const rainfallDisplay = document.getElementById('rainfall');

// Function to update hidden input fields with weather data
function updateWeatherData(temperature, humidity, weatherCondition, rainfall) {
    document.getElementById('temperature').value = temperature;
    document.getElementById('humidity').value = humidity;
    document.getElementById('weather-condition').value = weatherCondition;
    document.getElementById('rainfall').value = rainfall;
}

async function getData(lat, long) {
    const response = await fetch(`http://api.weatherapi.com/v1/current.json?key=930971becfa941f882053344220412&q=${lat},${long}&aqi=yes`);
    const data = await response.json();
    return data;
}

async function gotLocation(position) {
    const data = await getData(position.coords.latitude, position.coords.longitude);
    const current = data.current;

    const temperature = current.temp_c;
    const humidity = current.humidity;
    const weatherCondition = current.condition.text;
    const rainfall = current.precip_mm;

    // Update display
    temperatureDisplay.textContent = temperature + "°C";
    humidityDisplay.textContent = humidity + "%";
    weatherConditionDisplay.textContent = weatherCondition;
    rainfallDisplay.textContent = rainfall + " mm";

    // Update hidden input fields
    updateWeatherData(temperature, humidity, weatherCondition, rainfall);
}

function failedToGet(error) {
    console.log('There was some issue:', error.message);
}

button.addEventListener('click', async () => {
    navigator.geolocation.getCurrentPosition(gotLocation, failedToGet);
});

document.getElementById("landAreaDropdown").addEventListener("change", function() {
    var selectedArea = this.value;
    console.log("Selected area: " + selectedArea + " square meters");
});
