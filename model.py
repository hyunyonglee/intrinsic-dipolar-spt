# Copyright 2024 Hyun-Yong Lee

from tenpy.models.lattice import Chain, Lattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.tools.params import Config
from tenpy.networks.site import BosonSite
import numpy as np
import sym_sites
__all__ = ['IntrinsicDipolarSPT']



class Dipolar_Fermi_Hubbard_Ising( CouplingModel, MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "DIPOLAR_FERMI_HUBBARD")
        L = model_params.get('L', 1)
        t = model_params.get('t', 1.)
        tp = model_params.get('tp', 0.)
        U = model_params.get('U', 0.)
        mu = model_params.get('mu', 0.)
        J = model_params.get('J', 0.)
        h = model_params.get('h', 0.)
        Symmetries = model_params.get('Symmetries', {'N': False, 'Sz': False, 'D': False})
        
        sites = [ sym_sites.SpinHalfFermionSite_DipoleSymmetry(cons_N=Symmetries['N'], cons_Sz=Symmetries['Sz'], cons_D=Symmetries['D'], x=x) for x in range(L) ]
        for x in range(L):
            sites[x].multiply_operators(['Cu','Cd'])
            sites[x].multiply_operators(['Cd','Cu'])
            sites[x].multiply_operators(['Cdd','Cdu'])
            sites[x].multiply_operators(['Cdu','Cdd'])
            
        lat = Lattice([1], sites, order='default', bc='open', bc_MPS='finite', basis=[[L]], positions=np.array([range(L)]).transpose())

        CouplingModel.__init__(self, lat)

        # Ising term
        for x in range(L-1):
            self.add_multi_coupling( J, [('Sz', 0, x), ('Sz', 0, x+1)])

        # 3-site hopping
        for x in range(L-2):
            self.add_multi_coupling( -t, [('Cdu', 0, x), ('Cu Cd', 0, x+1), ('Cdd', 0, x+2)])
            self.add_multi_coupling( -t, [('Cdd', 0, x), ('Cd Cu', 0, x+1), ('Cdu', 0, x+2)])
        
            self.add_multi_coupling( -t, [('Cd', 0, x+2), ('Cdd Cdu', 0, x+1), ('Cu', 0, x)])
            self.add_multi_coupling( -t, [('Cu', 0, x+2), ('Cdu Cdd', 0, x+1), ('Cd', 0, x)])

        # 4-site hopping
        for x in range(L-3):
            self.add_multi_coupling( -tp, [('Cdu', 0, x), ('Cu', 0, x+1), ('Cd', 0, x+2), ('Cdd', 0, x+3)])
            self.add_multi_coupling( -tp, [('Cdd', 0, x), ('Cd', 0, x+1), ('Cu', 0, x+2), ('Cdu', 0, x+3)])

            self.add_multi_coupling( -tp, [('Cd', 0, x+3), ('Cdd', 0, x+2), ('Cdu', 0, x+1), ('Cu', 0, x)])
            self.add_multi_coupling( -tp, [('Cu', 0, x+3), ('Cdu', 0, x+2), ('Cdd', 0, x+1), ('Cd', 0, x)])
        
        # On-site
        for x in range(L):
            self.add_onsite( U, x, 'NuNd')
            self.add_onsite( -mu, x, 'Ntot')
            self.add_onsite( -h, x, 'Sx')
            
        MPOModel.__init__(self, lat, self.calc_H_MPO())