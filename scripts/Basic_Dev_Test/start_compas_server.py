from compas_cloud.server import CompasServerProtocol
try:
    from compas.data import DataEncoder
    from compas.data import DataDecoder
except ImportError:
    from compas.utilities import DataEncoder
    from compas.utilities import DataDecoder
import asyncio
from autobahn.asyncio.websocket import WebSocketServerFactory

factory = WebSocketServerFactory()
factory.protocol = CompasServerProtocol

# Get the local IP address
# This assumes you have an internet access, and that there is no local proxy.
try:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP = s.getsockname()[0]
    s.close()
except Exception:
    print("Could not get local IP address via accessing 8.8.8.8, using localhost")
    IP = '127.0.0.1'

PORT = 9009

loop = asyncio.get_event_loop()
coro = loop.create_server(factory, IP, PORT)
server = loop.run_until_complete(coro)
print("starting compas_cloud server")
print("Listening at %s:%s" % (IP, PORT))

try:
    loop.run_forever()
except KeyboardInterrupt:
    pass
finally:
    print("shutting down server")
    server.close()
    loop.close()
