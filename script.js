const formPrediksi = document.getElementById("form-prediksi");
const pilihanModel = document.querySelectorAll('input[name="model"]');
const inputFitur = document.querySelectorAll("#form-prediksi input");
const infoModel = document.getElementById("info-model");
const teksSpesies = document.getElementById("hasil-spesies");
const teksKeterangan = document.getElementById("hasil-keterangan");

const daftarSpesies = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
const petunjuk = "Isi ukuran bunga, lalu klik Prediksi";

let modelKnn;
let modelNb;

try {
  const responKnn = await fetch("model_knn.json", { cache: "no-cache" });
  modelKnn = await responKnn.json();

  const responNb = await fetch("model_nb.json", { cache: "no-cache" });
  modelNb = await responNb.json();
} catch {
  teksKeterangan.textContent = "File model_knn.json tidak ditemukan";
  teksKeterangan.classList.add("error");
}

const { k, data_min, data_max, x_train, y_train } = modelKnn;
const { gaussian } = modelNb;

const modelDipilih = () => {
  return document.querySelector('input[name="model"]:checked').value;
};

const singkatNama = (spesies) => {
  return spesies.replace("Iris-", "");
};

const tampilkanInfoModel = () => {
  const model = modelDipilih();
  const dataModel = model === "knn" ? modelKnn : modelNb;
  const jumlahData = dataModel.jumlah_train + dataModel.jumlah_test;
  const persenTest = Math.round((dataModel.jumlah_test / jumlahData) * 100);
  const persenTrain = 100 - persenTest;
  const infoK = model === "knn" ? ` (K = ${k})` : "";

  infoModel.textContent = `Train : Test = ${persenTrain}% : ${persenTest}%${infoK}`;
};

tampilkanInfoModel();

// KNN
const skalakan = (fitur) => {
  return fitur.map(
    (nilai, i) => (nilai - data_min[i]) / (data_max[i] - data_min[i]),
  );
};

const hitungJarak = (titikA, titikB) => {
  let jumlah = 0;
  for (let i = 0; i < titikA.length; i++) {
    jumlah += (titikA[i] - titikB[i]) ** 2;
  }
  return Math.sqrt(jumlah);
};

const tetanggaTerdekat = (fiturSkala) => {
  const daftarJarak = x_train.map((dataTrain, i) => ({
    jarak: hitungJarak(dataTrain, fiturSkala),
    spesies: y_train[i],
  }));
  daftarJarak.sort((a, b) => a.jarak - b.jarak);
  return daftarJarak.slice(0, k);
};

const jumlahPerSpesies = (daftarTetangga) => {
  const jumlah = {};
  for (const spesies of daftarSpesies) {
    jumlah[spesies] = 0;
  }
  for (const tetangga of daftarTetangga) {
    jumlah[tetangga.spesies] += 1;
  }
  return jumlah;
};

const spesiesTerbanyak = (jumlah) => {
  let terbanyak = daftarSpesies[0];
  for (const spesies of daftarSpesies) {
    if (jumlah[spesies] > jumlah[terbanyak]) {
      terbanyak = spesies;
    }
  }
  return terbanyak;
};

const prediksiKnn = (fitur) => {
  const fiturSkala = skalakan(fitur);
  const daftarTetangga = tetanggaTerdekat(fiturSkala);
  const jumlah = jumlahPerSpesies(daftarTetangga);
  const spesies = spesiesTerbanyak(jumlah);

  const daftarJumlah = [];
  for (const nama of daftarSpesies) {
    daftarJumlah.push(`${jumlah[nama]} ${singkatNama(nama)}`);
  }

  return {
    spesies,
    keterangan: `Dari ${k} tetangga terdekat:\n${daftarJumlah.join(", ")}`,
  };
};

// Naive Bayes
const skorSatuUkuran = (nilai, rataRata, varian) => {
  const selisihKuadrat = (nilai - rataRata) ** 2;
  return -0.5 * Math.log(2 * Math.PI * varian) - selisihKuadrat / (2 * varian);
};

const skorGaussian = (fitur) => {
  const daftarSkor = [];
  for (let s = 0; s < daftarSpesies.length; s++) {
    let skor = Math.log(gaussian.class_prior[s]);
    for (let u = 0; u < fitur.length; u++) {
      skor += skorSatuUkuran(fitur[u], gaussian.theta[s][u], gaussian.var[s][u])
    }
    daftarSkor.push(skor);
  }
  return daftarSkor;
};

const ubahKePeluang = (daftarSkor) => {
  const nilaiPeluang = daftarSkor.map((skor) => Math.exp(skor));
  let totalPeluang = 0;
  for (let i = 0; i < nilaiPeluang.length; i++) {
    totalPeluang += nilaiPeluang[i];
  }
  return nilaiPeluang.map((nilai) => nilai / totalPeluang);
}

const prediksiGaussian = (fitur) => {
  const daftarSkor = skorGaussian(fitur)
  const daftarPeluang = ubahKePeluang(daftarSkor)

  let indeksTerbesar = 0;
  for (let i = 1; i < daftarPeluang.length; i++) {
    if (daftarPeluang[i] > daftarPeluang[indeksTerbesar]) {
      indeksTerbesar = i;
    }
  }

  const rincianPeluang = []
  for (let i = 0; i < daftarSpesies.length; i++) {
    const persen = (daftarPeluang[i] * 100).toFixed(1);
    rincianPeluang.push(`${persen}% ${singkatNama(daftarSpesies[i])}`)
  }

  return {
    spesies: daftarSpesies[indeksTerbesar],
    keterangan: `Peluang tiap spesies:\n${rincianPeluang.join(", ")}`,
  };
};

// Prediksi
const prediksiSesuaiModel = (fitur) => {
  if (modelDipilih() === "knn") {
    return prediksiKnn(fitur);
  }
  return prediksiGaussian(fitur);
};

formPrediksi.addEventListener("submit", (e) => {
  e.preventDefault();

  const fitur = [];
  for (const input of inputFitur) {
    fitur.push(parseFloat(input.value));
  }

  const hasil = prediksiSesuaiModel(fitur);

  teksSpesies.textContent = singkatNama(hasil.spesies);
  teksKeterangan.textContent = hasil.keterangan;
});

// Perbarui Info
for (const pilihan of pilihanModel) {
  pilihan.addEventListener("change", () => {
    tampilkanInfoModel();
    teksSpesies.textContent = "";
    teksKeterangan.textContent = petunjuk;
  });
}