from distutils.core import setup
setup(
  name = 'ElM2D',        
  packages = ['ElM2D'],  
  version = '0.1.9',      
  license='GPL3',       
  description = 'A high performance mapping class to embed large datasets of ionic compositions with respect to the ElMD metric.',  
  author = 'Cameron Hagreaves',              
  author_email = 'cameron.h@rgreaves.me.uk', 
  url = 'https://github.com/lrcfmd/ElM2D/',   
  download_url = 'https://github.com/lrcfmd/ElM2D/archive/0.1.9.tar.gz',    
  keywords = ['ChemInformatics', 'Materials Science', 'Machine Learning', 'Materials Representation'],   
  install_requires=[            
          'cython',
          'numba',
          'numpy',
          'pandas',
          'tqdm',
          'scipy',
          'plotly',
          'umap-learn'
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',  
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3) ',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
)