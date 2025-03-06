$(document).ready(function () {

    // Display Speak Message
    function DisplayMessage(message) {
        $(".siri-message li:first").text(message);
        $('.siri-message').textillate('start');
    }

    // Expose DisplayMessage to Python
    eel.expose(DisplayMessage);

    // Display hood
    function ShowHood() {
        $("#Oval").show();
        $("#SiriWave").hide();
    }

    // Expose ShowHood to Python
    eel.expose(ShowHood);

    // Send message to chat box
    function senderText(message) {
        var chatBox = document.getElementById("chat-canvas-body");
        if (message.trim() !== "") {
            chatBox.innerHTML += `<div class="row justify-content-end mb-4">
                <div class="width-size">
                    <div class="sender_message">${message}</div>
                </div>
            </div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    }

    // Expose senderText to Python
    eel.expose(senderText);

    // Receive message in chat box
    function receiverText(message) {
        var chatBox = document.getElementById("chat-canvas-body");
        if (message.trim() !== "") {
            chatBox.innerHTML += `<div class="row justify-content-start mb-4">
                <div class="width-size">
                    <div class="receiver_message">${message}</div>
                </div>
            </div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    }

    // Expose receiverText to Python
    eel.expose(receiverText);

});