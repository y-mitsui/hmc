from distutils.core import setup, Extension

module1 = Extension('pmcmc',
                    include_dirs = ['/usr/include/glib-2.0','/usr/lib/x86_64-linux-gnu/glib-2.0/include','/usr/include/atlas','/usr/include/apr-1.0','/usr/include/libxml2'],
                    libraries = ['m','gsl','gslcblas','pthread'],
                    library_dirs = ['/usr/local/lib','/usr/lib/x86_64-linux-gnu'],
                    sources = ['pmcmc.c','mnorm.c'])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])

