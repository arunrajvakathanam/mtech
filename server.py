import socket

# Server configuration
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 12345

def main():
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Bind the socket to the address and port
        server_socket.bind((SERVER_HOST, SERVER_PORT))

        # Listen for incoming connections
        server_socket.listen(5)
        print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")

        while True:
            # Accept a new connection
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")

            # Send two numbers to the client
            #num1 = 5
            #num2 = 7
            #Enter the number in manually
            num1= int(input("Enter the first number : "))
            num2= int(input("Enter the Second Number : "))
            client_socket.sendall(f"{num1} {num2}".encode())

            # Receive the sum from the client
            sum_received = client_socket.recv(1024).decode()
            print(f"Sum received from client: {sum_received}")

            # Close the connection
            client_socket.close()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the server socket
        server_socket.close()

if __name__ == "__main__":
    main()
