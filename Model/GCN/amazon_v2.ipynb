{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# GCN 모델을 이용한 Graph Data training\n"
      ],
      "metadata": {
        "id": "OI7VEPHMEOm_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install torch_geometric"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VpnlaKugC61N",
        "outputId": "7927b24b-2df8-4bec-f3bb-cae7a328e2c7"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting torch_geometric\n",
            "  Downloading torch_geometric-2.4.0-py3-none-any.whl (1.0 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.0/1.0 MB\u001b[0m \u001b[31m10.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (4.66.1)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (1.23.5)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (1.11.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (3.1.2)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (2.31.0)\n",
            "Requirement already satisfied: pyparsing in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (3.1.1)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (1.2.2)\n",
            "Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.10/dist-packages (from torch_geometric) (5.9.5)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch_geometric) (2.1.3)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torch_geometric) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torch_geometric) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torch_geometric) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torch_geometric) (2023.11.17)\n",
            "Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->torch_geometric) (1.3.2)\n",
            "Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->torch_geometric) (3.2.0)\n",
            "Installing collected packages: torch_geometric\n",
            "Successfully installed torch_geometric-2.4.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 1) 필요한 라이브러리 선언"
      ],
      "metadata": {
        "id": "9oFXeyGTBGME"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import time\n",
        "import torch\n",
        "import torch_geometric.transforms as T\n",
        "import torch.nn as nn\n",
        "from torch_geometric.nn import GCNConv\n",
        "import torch.nn.functional as F"
      ],
      "metadata": {
        "id": "nKv3SHpjJbP5"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 2) 현재 시간을 기록하여 시간 측정 시작\n",
        "\n"
      ],
      "metadata": {
        "id": "StM3yXtOAVrV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "start_time = time.time()"
      ],
      "metadata": {
        "id": "QKD_IzgRAkw3"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 3) 장치를 \"cpu\"로 설정"
      ],
      "metadata": {
        "id": "w7UhKXKUAl79"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "device = \"cpu\""
      ],
      "metadata": {
        "id": "9xd4IlmaAq1i"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 4) Dataset 준비 (Amazon)"
      ],
      "metadata": {
        "id": "XLMfMzSlAuYX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from torch_geometric.datasets import Amazon\n",
        "\n",
        "dataset = Amazon(root='./Amazon_data', name='computers')\n",
        "graph = dataset[0]\n",
        "split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)\n",
        "graph = split(graph)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZixZRST-Ay_o",
        "outputId": "01c98baf-c0ef-44e4-f20b-48d150ffcf1a"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Downloading https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_computers.npz\n",
            "Processing...\n",
            "Done!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 5) 모델 구축: GCN"
      ],
      "metadata": {
        "id": "0LtPqk_FA1Mv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class GCN(torch.nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.conv1 = GCNConv(dataset.num_node_features, 16)\n",
        "        self.conv2 = GCNConv(16, dataset.num_classes)\n",
        "\n",
        "    def forward(self, data):\n",
        "        x, edge_index = data.x, data.edge_index\n",
        "        x = self.conv1(x, edge_index)\n",
        "        x = F.relu(x)\n",
        "        output = self.conv2(x, edge_index)\n",
        "        return output"
      ],
      "metadata": {
        "id": "5NPzO_NgA9vG"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 6) GCN 모델을 학습하는 함수 정의\n",
        "입력으로는 모델, 그래프 데이터, 옵티마이저, 손실함수, 에폭 횟수가 주어짐"
      ],
      "metadata": {
        "id": "iJ-tu5iVA_8D"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):\n",
        "    # 에폭 횟수만큼 학습 반복\n",
        "    for epoch in range(1, n_epochs + 1):\n",
        "        model.train() # 모델을 학습 상태로 전환\n",
        "        optimizer.zero_grad() # 그래디언트 초기화\n",
        "        out = model(graph) # out :예측값\n",
        "        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask]) # loss 계산\n",
        "        loss.backward() # Backpropagation 수행\n",
        "        optimizer.step() # 파라미터 업데이트\n",
        "\n",
        "        pred = out.argmax(dim=1) #out에서 가장 높은 값을 가지는 인덱스를 예측값으로 사용\n",
        "        acc = eval_node_classifier(model, graph, graph.val_mask) # 모델 성능 검증\n",
        "\n",
        "        if epoch % 10 == 0:\n",
        "            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')\n",
        "\n",
        "    return model"
      ],
      "metadata": {
        "id": "CXng3jJhBtKL"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 7) 노드 분류 모델의 성능을 평가하는 함수 정의\n",
        "입력으로는 모델, 그래프, 마스크가 주어짐"
      ],
      "metadata": {
        "id": "DK4b-QGcBvJX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def eval_node_classifier(model, graph, mask):\n",
        "\n",
        "    model.eval() # 모델을 평가 모드로 전환\n",
        "    # 모델의 출력 계산 -> argmax함수를 사용하여 출력 텐서에서 각 노드의 예측 클래스를 결정\n",
        "    pred = model(graph).argmax(dim=1)\n",
        "    # 예측된 클래스와 그래프의 실제 클래스를 비교하여 정확하게 분류된 노드의 수 계산\n",
        "    correct = (pred[mask] == graph.y[mask]).sum()\n",
        "    #print(pred[mask] + graph.y[mask])\n",
        "\n",
        "    # 정확도 계산\n",
        "    acc = int(correct) / int(mask.sum())\n",
        "\n",
        "    return acc"
      ],
      "metadata": {
        "id": "JPO5wL3yB6ZC"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 8) 모델 훈련"
      ],
      "metadata": {
        "id": "q0pYsDhNB7SI"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mG_cPxKOCa6_",
        "outputId": "0c2420e7-4333-4269-d191-7cb4278b4744"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch: 010, Train Loss: 1.927, Val Acc: 0.436\n",
            "Epoch: 020, Train Loss: 1.704, Val Acc: 0.463\n",
            "Epoch: 030, Train Loss: 1.535, Val Acc: 0.492\n",
            "Epoch: 040, Train Loss: 1.402, Val Acc: 0.500\n",
            "Epoch: 050, Train Loss: 1.213, Val Acc: 0.711\n",
            "Epoch: 060, Train Loss: 0.986, Val Acc: 0.742\n",
            "Epoch: 070, Train Loss: 0.819, Val Acc: 0.750\n",
            "Epoch: 080, Train Loss: 0.696, Val Acc: 0.812\n",
            "Epoch: 090, Train Loss: 0.601, Val Acc: 0.841\n",
            "Epoch: 100, Train Loss: 0.539, Val Acc: 0.851\n",
            "Epoch: 110, Train Loss: 0.498, Val Acc: 0.856\n",
            "Epoch: 120, Train Loss: 0.467, Val Acc: 0.863\n",
            "Epoch: 130, Train Loss: 0.444, Val Acc: 0.868\n",
            "Epoch: 140, Train Loss: 0.426, Val Acc: 0.870\n",
            "Epoch: 150, Train Loss: 0.410, Val Acc: 0.873\n",
            "Epoch: 160, Train Loss: 0.399, Val Acc: 0.875\n",
            "Epoch: 170, Train Loss: 0.388, Val Acc: 0.876\n",
            "Epoch: 180, Train Loss: 0.381, Val Acc: 0.876\n",
            "Epoch: 190, Train Loss: 0.371, Val Acc: 0.881\n",
            "Epoch: 200, Train Loss: 0.365, Val Acc: 0.878\n"
          ]
        }
      ],
      "source": [
        "gcn = GCN().to(device)\n",
        "optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "gcn = train_node_classifier(gcn, graph, optimizer_gcn, criterion)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 9) 결과 분석 및 소요시간 측정"
      ],
      "metadata": {
        "id": "l2mA6cUnC_Rl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "test_acc = eval_node_classifier(gcn, graph, graph.test_mask)\n",
        "print(f'Test Acc: {test_acc:.3f}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OcR5ZVV_C-xQ",
        "outputId": "02d1a473-194c-4565-c31e-0d92b9b2b208"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Test Acc: 0.893\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 시간 측정 종료\n",
        "end_time = time.time()\n",
        "print(\"총 소요 시간: %.3f초\" %(end_time - start_time))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rbu9Zm7wCsGz",
        "outputId": "0c8378b6-e448-4b63-c0b9-869b8ef10db5"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "총 소요 시간: 91.306초\n"
          ]
        }
      ]
    }
  ]
}
