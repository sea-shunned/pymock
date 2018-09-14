import subprocess

def run_exp(servers, l_ranges):
    # Loop over server/setup info
    for server_name, mut_method in servers:
        # Special case - no parameter searching
        if mut_method == "original":
            commands = ['python run_mock.py -runs 20 --synthdata "*_9*"']
        else:
            # generate as many commands as params we're varying
            commands = ['python run_mock.py -runs 20 --synthdata "*_9*"']*len(l_ranges)

            # Loop over these and customise the commands for relevant values
            for l_val, command in zip(l_ranges, commands):
                command += f" -m {mut_method} --L_comp {l_val} -e l{l_val}/{mut_method}"

        # Join commands so they are run sequentially (using &&)
        pyth_command = " && ".join(commands)

        # Construct the entire thing, with the required constant at start
        # base_cmd = [f"ssh -t shandc@{server_name} 'cd /local/home/shandc/PyMOCK && git pull && source activate py3env && {pyth_command}'"]

        base_cmd = [
            "ssh",
            "-t",
            f"shandc@{server_name}",
            "cd /local/home/shandc/PyMOCK",
            "&&",
            "source activate py3env",
            "&&",
            f"{pyth_command}"
        ]        
        ### Notes for future self
        # I don't think that screen is actually needed as a command if we create a screen here

        with open("output.txt","wb") as file:
            p = subprocess.Popen(base_cmd, stdout=file)


# check why folder is test_data_2
# sort out the l_star thing
# collect/check results on ds1
# then deployyyyyy
# then work on graphs again

if __name__ == '__main__':
    # servers = ['datascience1', 'datascience2']
    
    # for server in servers:
    #     # base_cmd = f"ssh -t shandc@{server} 'cd /local/home/shandc/PyMOCK && hostname && source activate py3env && python run_mock.py -v'"

    #     base_cmd = [
    #         "ssh",
    #         "-t",
    #         f"shandc@{server}",
    #         "cd /local/home/shandc/PyMOCK",
    #         "&&",
    #         "source activate py3env",
    #         "&&",
    #         "python run_mock.py -v"
    #     ]

    #     # with open("output.txt", "wb") as file:
    #     #     p = subprocess.Popen(base_cmd, stdout=file)
    #     p = subprocess.Popen(base_cmd)

    servers = [
        ("datascience1", "original"),
        ("datascience2", "centroid")
        # ("datascience3", "neighbour")
    ]
    l_ranges = range(1, 6)
    run_exp(servers, l_ranges)
