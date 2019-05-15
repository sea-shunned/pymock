import rpy2
import readline # fixes problem with rpy2
import plots

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

eaf = importr('eaf', lib_loc="/home/cshand/R/x86_64-pc-linux-gnu-library/3.4")

robjects.r('''
    plotEAF <- function(fit_L, fit_R, ari_L, ari_R, label_L, label_R, switchcols=TRUE){
        data_L <-read.table(fit_L, sep=",")
        data_R <-read.table(fit_R, sep=",")

        if (switchcols){
            data_L <- data_L[c(2,1,3)]
            data_R <- data_R[c(2,1,3)]
        }

        ari_L = read.table(ari_L, sep=",")
        ari_R = read.table(ari_R, sep=",")

        max_ari_L = max(ari_L)
        max_ari_R = max(ari_R)

        print(max_ari_L)
        print(max_ari_R)

        # Adjust the column name so it works properly
        colnames(data_L)[3] <- "set"
        colnames(data_R)[3] <- "set"

        # Extract the fitness values for this ARI from the correct dataset
        if (max_ari_L >= max_ari_R){
            indiv_num = which.max(apply(ari_L, MARGIN = 1, max))
            run_num = unname(which.max(apply(ari_L, MARGIN = 2, max)))

            fit_vals = as.numeric(data_L[which(data_L$set==run_num),][indiv_num,1:2])
            
        } else
            indiv_num = which.max(apply(ari_R, MARGIN = 1, max))
            run_num = unname(which.max(apply(ari_R, MARGIN = 2, max)))
            
            fit_vals = as.numeric(data_R[which(data_R$set==run_num),][indiv_num,1:2])                
        print(fit_vals)

        eafdiffplot(data_L, data_R, 
            ylab="Intracluster Variance", xlab="Connectivity",
            title.left = label_L, title.right = label_R,
            type="area",
            left.panel.last = { points(fit_vals[1], fit_vals[2], pch=23, cex=1.5, bg='red') },
            right.panel.last = { points(fit_vals[1], fit_vals[2], pch=23, cex=1.5, bg='red') }
        )
    }
''')

def plot_eaf():
    pass

    # Bring everything together (from the main_block) into this function
    # then integrate it with plots and params
    # might want to do both simultaneously to avoid duplication

if __name__ == '__main__':
    base_folder = plots.base_res_folder("mut_ops")
    print(base_folder)

    mut_method = "neighbour"
    Lval = "L5"

    ari_L = plots.get_fpaths(base_folder / "orig", "*20_80*ari*")[0]
    fit_L = plots.get_fpaths(base_folder / "orig", "*20_80*fitness*")[0]
    print(ari_L)
    print(fit_L)

    ari_R = plots.get_fpaths(base_folder / mut_method / Lval, "*20_80*ari*")[0]
    fit_R = plots.get_fpaths(base_folder / mut_method / Lval, "*20_80*fitness*")[0]
    print(ari_R)
    print(fit_R)

    # Need as params (if stmt) in plots
    label_L = "orig"
    label_R = mut_method+"-"+Lval

    ploteaf = robjects.r['plotEAF']

    # Convert to str as Posix Paths
    ploteaf(str(fit_L), str(fit_R), str(ari_L), str(ari_R), label_L, label_R)

    # To save the generated graph
    # Need to set fname dynamically
    grdevices = importr('grDevices')
    grdevices.dev_copy2eps(file="eaf.eps", onefile=True)

    # Now need to save the plot