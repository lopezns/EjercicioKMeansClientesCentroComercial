from flask import Flask, render_template, request
import Clustering


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dataset/")
def dataset():
    info = Clustering.ObtenerInformacionDataset()
    return render_template("dataset.html", info=info)


@app.route("/conceptos/")
def conceptos():
    return render_template("conceptos.html")


@app.route("/modelo/", methods=["GET", "POST"])
def modelo():
    nclusters = request.form.get("nclusters", 5, type=int)
    info = Clustering.RealizarClustering(nclusters)
    return render_template("modelo.html", info=info, nclusters=nclusters)


@app.route("/Cluster/")
def cluster():
    info = Clustering.RealizarClustering()
    return render_template("ClusterResults.html", resultados=info["resumenClusters"])


@app.route("/interpretacion/")
def interpretacion():
    info = Clustering.RealizarClustering()
    return render_template("interpretacion.html", info=info)


if __name__ == "__main__":
    app.run(debug=True)
