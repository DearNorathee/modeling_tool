from setuptools import setup, find_packages

setup(
    author= "Dear Norathee",
    description="the extension of sklearn to help the your modeling code becomes more concise with common useful tool for modeling",
    name="modeling_tool",
    version="0.1.3",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "os_toolkit>=0.1.1",
        "dataframe_short>=0.1.6",
        "python_wizard>=0.1.2",
        "inspect_py>=0.1.1",
        "py_string_tool>=0.1.3",

        "scikit-learn",
        "lightgbm",
        "xgboost",
        "imblearn",
        "playsound",
        "seaborn",


        ],
    python_requires='>=3.10.0',
    extras_require={
        'full': ['torch>=1.0']  # Optional torch dependency
    },
    # example
    # install_requires=['pandas>=1.0',
    # 'scipy==1.1',
    # 'matplotlib>=2.2.1,<3'],
    

)