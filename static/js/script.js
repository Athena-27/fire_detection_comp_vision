function fetchStatus() {
    $.ajax({
        url: "/fire_status",
        type: "GET",
        success: function(data) {
            if (data.fire_detected) {
                $('#status').text(`Fire Detected: ${data.fire_probability.toFixed(2)}%`);
                $('#status').removeClass('no-fire').addClass('fire-detected');
            } else {
                $('#status').text('No Fire Detected');
                $('#status').removeClass('fire-detected').addClass('no-fire');
            }
        }
    });
}

$(document).ready(function() {
    setInterval(fetchStatus, 1000);
});
