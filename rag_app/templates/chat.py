chat_html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <title></title>
  </head>
  <body>
    <!--widget-websocket="ws://localhost:8000/chat/agent"-->
    <webchat-widget
      widget-websocket="ws://localhost:8000/chat/agent"
      widget-color="#47A7F6"
      widget-chat-avatar="https://icon-library.com/images/bot-icon/bot-icon-1.jpg"
      widget-user-avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWR4hHJTiikyzCv6nc5OAkHPIHMD-ESsP-LFEaY2vVIjV6wqCt&s"
      widget-header="Bot"
      widget-subheader="Online"
      widget-placeholder="Send a message"
      widget-position="true"
      widget-on="false"
    >
    </webchat-widget>
    <script src="https://webchat-widget.pages.dev/static/js/main.js"></script>
  </body>
</html>
"""