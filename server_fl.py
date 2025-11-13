import flwr as fl

if __name__ == "__main__":
    # Define strategy
    strategy = fl.server.strategy.FedAvg()

    # Define proper ServerConfig (instead of dictionary)
    server_config = fl.server.ServerConfig(num_rounds=3)

    # Start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy,
    )
