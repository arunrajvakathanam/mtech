import socket

# Server configuration
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 12345
BUFFER_SIZE = 1024

def main():
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect the socket to the server
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        print(f"Connected to {SERVER_HOST}:{SERVER_PORT}")

        # Receive two numbers from the server
        numbers = client_socket.recv(BUFFER_SIZE).decode()
        num1, num2 = map(int, numbers.split())

        # Calculate the sum
        sum_result = int(num1) + int(num2)
        print(sum_result)
        print(f"Received numbers from server: {num1}, {num2}")

        # Send the sum back to the server
        client_socket.sendall(str(sum_result).encode())

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the client socket
        client_socket.close()

if __name__ == "__main__":
    main()
