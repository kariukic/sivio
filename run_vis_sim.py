import os
import numpy as np
from casacore.tables import table

import vis_sim
import scint_equations as sqs

def generate_distribution(mean, sigma, size, dist):
    '''
    #Other distribution types
    if dist == "constant":
        return np.ones(size) * mean
    elif dist == "lognormal":
        return np.random.lognormal(loc=mean,
                                   sigma=sigma,
                                   size=size)
    '''
    if dist == "normal":
        return np.random.normal(loc=mean,
                                scale=sigma,
                                size=size)
    else:
        raise ValueError("Unrecognised/unimplemented distribution ({}) used.".format(dist))


N = 1000

ras = [3]#[4., 1] # generate_distribution(0., 4., N, "normal") #np.array([0,2])
decs = [-25]#[-28, -23., ] # generate_distribution(-27., 4., N, "normal") #np.array([-27.,-24.])
fluxes = [1]#[1, 2] # np.abs(generate_distribution(1., 3., N, "normal")) #np.array([1.,2.])
rdiffs = [1000] # diffractive scales.



def run_vis(mset, ras,decs,fluxes,rdiffs, clean_vis=False, offset=True, sc=False):
    for rd in rdiffs:
        os.system("mkdir %s" % (mset))
        os.system("cp -r 2sources_truesimvis.ms/* %s" % (mset))

        tbl = table(mset, readonly=False)
        vis_sim.simulate_vis(mset, tbl, ras, decs, fluxes, rd, clean_vis=clean_vis, offset=offset, scintillate=sc)
        tbl.close()

        imagename = mset.split('.')[0] + "_truevis"
        datacol='DATA'
        command = 'wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 -data-column %s %s'%(imagename, datacol, mset)
        os.system(command)

        if sc:
            imagename = mset.split('.')[0] + '_%srd_scint' % (int(rd/1000))
            command2 = 'wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 -data-column %s %s'%(imagename, 'SCINT_DATA', mset)
            os.system(command2)
        if offset:
            imagename = mset.split('.')[0] + '_offsetvis'
            command3 = 'wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 -data-column %s %s'%(imagename, 'OFFSET_DATA', mset)
            os.system(command3)

        
        #command2 = 'wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 -data-column OFFSET_DATA %s'%(imagename2, mset)

        
        #command1 = 'wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 -data-column DATA %s'%(imagename1, mset)
        

        #os.system(command1)
        #os.system(command2)
        #os.system('rm *dirty* *psf* *residual* *model*')

if __name__ == "__main__":
    #mset = '%skm_rdiff_2sources_ionosimvis.ms'% (int(rd/1000))
    mset = '2sources_offsettest.ms'
    run_vis(mset, ras, decs, fluxes, rdiffs, clean_vis=True, offset=True, sc=False)
    os.system('rm -r %s' % (mset))






    """
    col = 'RDIFF%sKM'%(int(rd/1000))
    if col not in tbl.colnames():
        coldmi = tbl.getdminfo('DATA')     # get dminfo of existing column
        coldmi["NAME"] = "tsm2"             # give it a unique name
        tbl.addcols(maketabdesc(makearrcoldesc(col, 0.+1j, shape=(224028, 768, 4), valuetype='dcomplex')), coldmi)
    """


    



