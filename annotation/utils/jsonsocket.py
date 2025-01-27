"""
This module provides functions to send and receive messages over a tcp socket using JSON serialization.
"""
import json

def receive_message(connection):
    """ Assuming that the first four bytes of the message denotes its length,
        will read an incoming tcp message from a socket connection
    """
    message_length_bytes = connection.recv(4)
    message_length = int.from_bytes(message_length_bytes, "big")
    message_body_bytes = connection.recv(message_length)
    if len(message_body_bytes) == 0:
        return None
    message = _parse_message(message_body_bytes)
    return message

def send_message(connection, body):
    """ Given a connected socket, will serialize the string value "body", send the length
        of the message in the first four bytes, and then sends the message
    """
    message = _format_message(body)
    connection.send(message)
    
def _parse_message(message_body_bytes):
    message_body = message_body_bytes.decode("UTF8")
    payload = json.loads(message_body)
    return payload
    
def _format_message(body):
    json_message = json.dumps(body)
    message_length = len(json_message)
    header_bytes = message_length.to_bytes(4, "big")
    message_bytes = json_message.encode("UTF8")
    message = header_bytes + message_bytes
    return message

def easy_send_message(connection, body, ack="annotation_response_ack"):
    """Sends a message to the remote server and waits for an acknowledgement message.

    Args:
        connection (socket.socket): The socket connection to the server.
        body (dict): The message to send to the server.
        ack (str, optional): The type of message to wait for as an acknowledgement. Defaults to "annotation_response_ack".

    Raises:
        ValueError: If the response message is not of the expected type.
        TimeoutError: If the server does not respond within the specified timeout.
    """
    send_message(connection, body)
    
    if ack is not None:
        
        connection.settimeout(5)
        try:
            response_message = receive_message(connection)
        except Exception as e:
            connection.settimeout(None)
            raise TimeoutError("Server did not respond within the specified timeout.")
            
        connection.settimeout(None)
        
        response_type = response_message.get("type")
        if response_type == 'disconnect':
            return
        
        if response_message.get("type") != ack:
            raise ValueError("Unexpected response message: " + str(response_message))
        
def easy_handshake(connection):
    """Sends a handshake message to the server and waits for a response.

    Args:
        connection (socket.socket): The socket connection to the server.

    Returns:
        Dict: The response message from the server. On error, returns an error message.
    """

    handshake_message = {"type": "handshake"}
    send_message(connection, handshake_message)
    connection.settimeout(5)
    try:
        response_message = receive_message(connection)
    except Exception as e:
        response_message = {"type": "error", "message": str(e)}
    connection.settimeout(None)  
    return response_message

def easy_disconnect(connection):
    """Sends a disconnect message to the server and waits for a response.

    Args:
        connection (socket.socket): The socket connection to the server.

    Returns:
        Dict: The response message from the server. On error, returns an error message.
    """
    response_message = None
    disconnect_message = {"type": "disconnect"}
    send_message(connection, disconnect_message)
    
    try:
        connection.settimeout(5)
        response_message = receive_message(connection)
        connection.settimeout(None)
        connection.close()
    except Exception as e:
        response_message = {"type": "error", "message": str(e)}
        connection.settimeout(None)
        
    return response_message