# Set your global username
git config --global user.name "davidsvaughn"

# Set your global email
git config --global user.email "davidsvaughn@gmail.com"

# To store credentials permanently (using git credential helper)
git config --global credential.helper store

Credentials will be stored in plaintext at ~/.git-credentials

--------------------------------------------------------------
dsv-llm-x1
---------------
export AZIP=azureuser@20.121.119.98
ssh -i $AZURE_EAST_PEM $AZIP
--------------------------------------------------------------

git clone https://github.com/davidsvaughn/baso
# git config --global credential.helper store
cd baso

virtualenv -p python3.10 venv && source venv/bin/activate
pip install --upgrade pip

pip install numpy pandas matplotlib scipy torch gpytorch botorvh


pip install spyder
# pip install spyder-kernels==3.0.*