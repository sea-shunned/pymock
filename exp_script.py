# Use a script to distribute experiments so server
# pssh isn't suitable

servers = [
    "datascience1",
    "datascience2",
    "datascience3",
    "datascience4"
]

# will need to set up keys to server to login

for server_name, command in zip(servers, commands):
    pass
    # ssh server_name screen command