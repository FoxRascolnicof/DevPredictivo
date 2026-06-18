"""
================================================================================
CHAKIPA — Sistema Inteligente para el Banco de Alimentos de Cajamarca
================================================================================
Metodología : CRISP-DM
Autor       : Brayan

Pipeline completo:
  0. Carga y limpieza de datos          (pandas)
  1. Módulo Predictivo de Demanda       (scikit-learn — RandomForestRegressor)
  2. Módulo de Optimización del Kit     (PuLP — PLI solver CBC)
  3. Módulo de Sustitución Inteligente  (NumPy — distancia euclidiana normalizada)

Librerías   : pandas, numpy, scikit-learn, pulp
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pulp


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CONSTANTES Y PARÁMETROS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = r"D:\Ingenieria_de_SistemasUPN\Ciclo_9_2026_1\MachineLearning\Semana_16\Crisp_Dm\BD_Completa.xlsx"

# ---------------------------------------------------------------------------
# Tabla de Composición de Alimentos del Perú (INS/MINSA — CENAN, 2017)
# ¡CORREGIDO! Nombres idénticos a las filas de la hoja 'Inventario' de BD_Completa.xlsx
# ---------------------------------------------------------------------------
TABLA_NUTRICIONAL: dict[str, dict] = {
    #  Categoría            kcal    prot(g)  carb(g)   gras(g)  fibra(g)
    "Aceite o grasas":   {"kcal": 8840, "proteina":   0, "carbohidrato":   0, "grasa": 1000, "fibra":   0},
    "Azúcar o dulce":    {"kcal": 3870, "proteina":   0, "carbohidrato": 999, "grasa":    0, "fibra":   0},
    "Fideo":             {"kcal": 3580, "proteina": 125, "carbohidrato": 738, "grasa":   16, "fibra":  25},
    "Panadería":         {"kcal": 2650, "proteina":  79, "carbohidrato": 519, "grasa":   34, "fibra":  26},
    "Menestras":         {"kcal": 3360, "proteina": 214, "carbohidrato": 616, "grasa":   12, "fibra": 155},
    "Carnes rojas":      {"kcal": 2180, "proteina": 206, "carbohidrato":   0, "grasa":  148, "fibra":   0},
    "Carnes blancas":    {"kcal": 1650, "proteina": 215, "carbohidrato":   0, "grasa":   87, "fibra":   0},
    "Lácteos":           {"kcal":  610, "proteina":  32, "carbohidrato":  48, "grasa":   36, "fibra":   0},
    "Salsas":            {"kcal":  800, "proteina":  20, "carbohidrato": 150, "grasa":   10, "fibra":  30},
    "Vegetales y hojas": {"kcal":  250, "proteina":  20, "carbohidrato":  55, "grasa":    3, "fibra":  35},
    "Otros almidones":   {"kcal": 3620, "proteina":  67, "carbohidrato": 794, "grasa":    9, "fibra":  55},
}

# ---------------------------------------------------------------------------
# Requerimientos nutricionales MÍNIMOS por kit completo (6 productos × 1 kg)
# ---------------------------------------------------------------------------
REQUERIMIENTOS_MINIMOS: dict[str, float] = {
    "kcal":         8000,   
    "proteina":      150,   
    "carbohidrato":  800,   
    "grasa":         100,   
    "fibra":          50,   
}

# Feriados oficiales del Perú
FERIADOS_PERU = pd.to_datetime([
    "2020-01-01","2020-04-09","2020-04-10","2020-05-01","2020-06-29",
    "2020-07-28","2020-07-29","2020-08-30","2020-10-08","2020-11-01",
    "2020-12-08","2020-12-25",
    "2021-01-01","2021-04-01","2021-04-02","2021-05-01","2021-06-29",
    "2021-07-28","2021-07-29","2021-08-30","2021-10-08","2021-11-01",
    "2021-12-08","2021-12-25",
    "2022-01-01","2022-04-14","2022-04-15","2022-05-01","2022-06-29",
    "2022-07-28","2022-07-29","2022-08-30","2022-10-08","2022-11-01",
    "2022-12-08","2022-12-25",
    "2023-01-01","2023-04-06","2023-04-07","2023-05-01","2023-06-29",
    "2023-07-28","2023-07-29","2023-08-30","2023-10-08","2023-11-01",
    "2023-12-08","2023-12-25",
    "2024-01-01","2024-03-28","2024-03-29","2024-05-01","2024-06-29",
    "2024-07-28","2024-07-29","2024-08-30","2024-10-08","2024-11-01",
    "2024-12-08","2024-12-25",
    "2025-01-01","2025-04-17","2025-04-18","2025-05-01","2025-06-29",
    "2025-07-28","2025-07-29","2025-08-30","2025-10-08","2025-11-01",
    "2025-12-08","2025-12-25",
    "2026-01-01","2026-04-02","2026-04-03",
])


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CARGA Y LIMPIEZA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_datos(ruta: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee BD_Completa.xlsx y procesa los despachos e inventarios.
    """
    sheets = pd.read_excel(ruta, sheet_name=None)

    # ── Reconstruir df_despachos ──────────────────────────────────────────────
    raw = sheets["Volúmenes DIstribuidos "]
    registros = []
    cur_nro, cur_fecha, cur_ben, cur_ninos, cur_bruto, cur_prods = (
        None, None, None, None, None, []
    )

    for _, fila in raw.iterrows():
        if pd.notna(fila["Nro."]) and pd.notna(fila["Fecha de despacho"]):
            if cur_nro is not None and cur_prods:
                registros.append({
                    "Nro":           cur_nro,
                    "Fecha":         cur_fecha,
                    "Beneficiarios": cur_ben,
                    "Ninos":         cur_ninos,
                    "Peso_Bruto_kg": cur_bruto,
                    "Productos":     cur_prods.copy(),
                })
            cur_nro   = fila["Nro."]
            cur_fecha = fila["Fecha de despacho"]
            cur_ben   = fila["Nº Beneficiarios"]
            cur_ninos = fila["Nº Niños"]
            cur_bruto = fila["Peso bruto total (kg)"]
            cur_prods = ([fila["Producto / Alimento"]]
                         if pd.notna(fila["Producto / Alimento"]) else [])
        elif pd.notna(fila["Producto / Alimento"]):
            cur_prods.append(fila["Producto / Alimento"])

    if cur_nro is not None and cur_prods:
        registros.append({
            "Nro": cur_nro, "Fecha": cur_fecha, "Beneficiarios": cur_ben,
            "Ninos": cur_ninos, "Peso_Bruto_kg": cur_bruto, "Productos": cur_prods,
        })

    df_despachos = (
        pd.DataFrame(registros)
        .assign(Fecha=lambda d: pd.to_datetime(d["Fecha"]))
        .dropna(subset=["Beneficiarios", "Fecha"])
        .sort_values("Fecha")
        .reset_index(drop=True)
    )

    # ── Stock actual real: Extrae el último estado de inventario reportado ────
    inv_raw = (
        sheets["Inventario"]
        .assign(Fecha=lambda d: pd.to_datetime(d["Fecha"]))
    )
    df_inventario = (
        inv_raw.sort_values("Fecha")
               .groupby("Categoría de Alimento", as_index=False)
               .last()[["Categoría de Alimento", "Stock Disponible (kg)"]]
               .rename(columns={
                   "Categoría de Alimento": "Categoria",
                   "Stock Disponible (kg)": "Stock_kg",
               })
    )

    return df_despachos, df_inventario


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — MÓDULO PREDICTIVO DE DEMANDA (RandomForestRegressor)
# ══════════════════════════════════════════════════════════════════════════════

def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el vector de características temporales e históricos para el modelo.
    """
    df = df.copy().sort_values("Fecha").reset_index(drop=True)

    df["Dia_Semana"]    = df["Fecha"].dt.dayofweek
    df["Mes"]           = df["Fecha"].dt.month
    df["Quincena"]      = (df["Fecha"].dt.day > 15).astype(int)
    df["Semana_Anio"]   = df["Fecha"].dt.isocalendar().week.astype(int)
    df["Trimestre"]     = df["Fecha"].dt.quarter
    df["Es_Feriado"]    = df["Fecha"].isin(FERIADOS_PERU).astype(int)
    df["Dias_Desde_Ini"] = (df["Fecha"] - df["Fecha"].min()).dt.days

    df = df.set_index("Fecha")
    serie = df["Beneficiarios"]
    df["Hist_Asistencia_7d"]  = serie.rolling("7D",  min_periods=1).mean().shift(1)
    df["Hist_Asistencia_30d"] = serie.rolling("30D", min_periods=1).mean().shift(1)
    df = df.reset_index()

    df["Lag_1"] = df["Beneficiarios"].shift(1)
    df["Lag_2"] = df["Beneficiarios"].shift(2)
    df["Lag_3"] = df["Beneficiarios"].shift(3)

    return df.dropna(subset=["Lag_1", "Lag_2", "Lag_3"]).reset_index(drop=True)


FEATURE_NAMES = [
    "Dia_Semana", "Mes", "Quincena", "Semana_Anio", "Trimestre",
    "Es_Feriado", "Dias_Desde_Ini",
    "Hist_Asistencia_7d", "Hist_Asistencia_30d",
    "Lag_1", "Lag_2", "Lag_3",
]


def entrenar_modelo_predictivo(
    df_features: pd.DataFrame,
) -> tuple[RandomForestRegressor, np.ndarray, np.ndarray, np.ndarray]:
    X = df_features[FEATURE_NAMES].values
    y = df_features["Beneficiarios"].values

    corte   = int(len(X) * 0.80)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    modelo = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    return modelo, X_test, y_test, y_pred


def evaluar_modelo(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2":   r2_score(y_test, y_pred),
    }


def predecir_beneficiarios(
    modelo: RandomForestRegressor,
    fecha_objetivo: pd.Timestamp,
    df_historico: pd.DataFrame,
) -> int:
    serie = (
        df_historico.set_index("Fecha")["Beneficiarios"]
                    .sort_index()
                    .dropna()
    )

    lag1 = float(serie.iloc[-1])
    lag2 = float(serie.iloc[-2])
    lag3 = float(serie.iloc[-3])

    umbral_7d  = fecha_objetivo - pd.Timedelta(days=7)
    umbral_30d = fecha_objetivo - pd.Timedelta(days=30)
    hist_7d  = float(serie[serie.index >= umbral_7d ].mean()
                     if serie[serie.index >= umbral_7d ].size > 0 else serie.mean())
    hist_30d = float(serie[serie.index >= umbral_30d].mean()
                     if serie[serie.index >= umbral_30d].size > 0 else serie.mean())

    x_nuevo = np.array([[
        fecha_objetivo.dayofweek,
        fecha_objetivo.month,
        int(fecha_objetivo.day > 15),
        fecha_objetivo.isocalendar()[1],
        fecha_objetivo.quarter,
        int(fecha_objetivo in FERIADOS_PERU),
        (fecha_objetivo - serie.index.min()).days,
        hist_7d,
        hist_30d,
        lag1,
        lag2,
        lag3,
    ]])

    pred = modelo.predict(x_nuevo)[0]
    return max(1, round(pred))


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — MÓDULO DE OPTIMIZACIÓN DEL KIT (PuLP — PLI)
# ══════════════════════════════════════════════════════════════════════════════

def optimizar_kit(
    beneficiarios_estimados: int,
    df_inventario: pd.DataFrame,
    tabla_nutricional: dict = TABLA_NUTRICIONAL,
    requerimientos: dict    = REQUERIMIENTOS_MINIMOS,
) -> dict:
    stock = dict(zip(df_inventario["Categoria"], df_inventario["Stock_kg"]))
    categorias = [c for c in tabla_nutricional if c in stock]

    max_kcal  = max(tabla_nutricional[c]["kcal"]     for c in categorias) or 1
    max_prot  = max(tabla_nutricional[c]["proteina"] for c in categorias) or 1
    max_fibra = max(tabla_nutricional[c]["fibra"]    for c in categorias) or 1

    prob = pulp.LpProblem("Chakipa_Kit_Optimization", pulp.LpMaximize)

    x = {
        c: pulp.LpVariable(f"x_{c.replace(' ', '_')}", cat="Binary")
        for c in categorias
    }

    # Maximizar balance nutricional equilibrado
    prob += pulp.lpSum(
        (
            0.40 * (tabla_nutricional[c]["kcal"]     / max_kcal)
          + 0.40 * (tabla_nutricional[c]["proteina"] / max_prot)
          + 0.20 * (tabla_nutricional[c]["fibra"]    / max_fibra)
        ) * x[c]
        for c in categorias
    )

    # Restricciones operativas del almacén
    prob += pulp.lpSum(x[c] for c in categorias) == 6, "Kit_6_alimentos"

    for c in categorias:
        prob += stock.get(c, 0) - beneficiarios_estimados * x[c] >= 0, \
               f"Stock_{c.replace(' ', '_')}"

    for nutriente, minimo in requerimientos.items():
        prob += (
            pulp.lpSum(tabla_nutricional[c][nutriente] * x[c] for c in categorias)
            >= minimo,
            f"Nutricion_{nutriente}",
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    estado         = pulp.LpStatus[prob.status]
    kit_elegido    = [c for c in categorias if pulp.value(x[c]) == 1]
    valor_objetivo = pulp.value(prob.objective) or 0.0

    cobertura = {}
    if kit_elegido:
        for nutriente, minimo in requerimientos.items():
            total = sum(tabla_nutricional[c][nutriente] for c in kit_elegido)
            cobertura[nutriente] = {
                "total":  round(total, 1),
                "minimo": minimo,
                "cumple": total >= minimo,
            }

    return {
        "kit":       kit_elegido,
        "estado":    estado,
        "puntaje":   round(valor_objetivo, 4),
        "nutricion": cobertura,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — MÓDULO DE SUSTITUCIÓN INTELIGENTE (NumPy)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_similitud_nutricional(
    alimento_objetivo: str,
    candidatos: list[str],
    tabla_nutricional: dict = TABLA_NUTRICIONAL,
) -> pd.DataFrame:
    NUTRIENTES = ["kcal", "proteina", "carbohidrato", "grasa", "fibra"]
    universo   = list(tabla_nutricional.keys())

    matriz = np.array(
        [[tabla_nutricional[a][n] for n in NUTRIENTES] for a in universo],
        dtype=float,
    )

    min_vals = matriz.min(axis=0)
    max_vals = matriz.max(axis=0)
    rango    = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1.0)
    matriz_norm = (matriz - min_vals) / rango

    idx_obj = universo.index(alimento_objetivo)
    v_obj   = matriz_norm[idx_obj]

    idx_cands = [universo.index(c) for c in candidatos if c in universo and c != alimento_objetivo]
    if not idx_cands:
        return pd.DataFrame(columns=["Alimento", "Distancia", "Similitud"])

    vectores_cands = matriz_norm[idx_cands]
    distancias     = np.linalg.norm(vectores_cands - v_obj, axis=1)
    similitudes    = 1.0 / (1.0 + distancias)

    nombres = [universo[i] for i in idx_cands]
    return (
        pd.DataFrame({"Alimento": nombres, "Distancia": distancias.round(4),
                      "Similitud": similitudes.round(4)})
        .sort_values("Similitud", ascending=False)
        .reset_index(drop=True)
    )


def activar_sustitucion(
    kit_propuesto: list[str],
    beneficiarios: int,
    df_inventario: pd.DataFrame,
    tabla_nutricional: dict = TABLA_NUTRICIONAL,
) -> dict[str, str | None]:
    stock = dict(zip(df_inventario["Categoria"], df_inventario["Stock_kg"]))
    sustituciones: dict[str, str | None] = {}

    for alimento in kit_propuesto:
        if stock.get(alimento, 0) < beneficiarios:
            candidatos = [
                c for c in tabla_nutricional
                if stock.get(c, 0) >= beneficiarios and c not in kit_propuesto
            ]
            if candidatos:
                df_sim   = calcular_similitud_nutricional(alimento, candidatos, tabla_nutricional)
                sustituto = df_sim.iloc[0]["Alimento"] if not df_sim.empty else None
            else:
                sustituto = None
            sustituciones[alimento] = sustituto

    return sustituciones


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def _sep(titulo: str = "") -> None:
    linea = "═" * 75
    print(f"\n{linea}")
    if titulo:
        print(f"  {titulo}")
        print(linea)


def ejecutar_pipeline(
    ruta_datos: Path          = DATA_PATH,
    fecha_objetivo: str | None = None,
) -> dict:
    # PASO 0: Carga y Limpieza
    _sep("PASO 0 — Carga y limpieza de datos")
    df_despachos, df_inventario = cargar_datos(ruta_datos)

    fecha_obj = (
        pd.Timestamp(fecha_objetivo)
        if fecha_objetivo
        else df_despachos["Fecha"].max() + pd.Timedelta(days=3)
    )

    print(f"  ✔ Despachos históricos : {len(df_despachos):>5} jornadas")
    print(f"  ✔ Categorías en stock  : {len(df_inventario):>5}")
    print(f"  ✔ Rango histórico      : {df_despachos['Fecha'].min().date()} → {df_despachos['Fecha'].max().date()}")
    print(f"  ✔ Fecha de proyección  : {fecha_obj.date()}")

    # PASO 1: Módulo Predictivo
    _sep("PASO 1 — Módulo Predictivo de Demanda (RandomForestRegressor)")
    df_feat = construir_features(df_despachos)
    print(f"  ✔ Dataset con features : {df_feat.shape[0]} muestras × {len(FEATURE_NAMES)} features")

    modelo, X_test, y_test, y_pred = entrenar_modelo_predictivo(df_feat)
    met = evaluar_modelo(y_test, y_pred)

    print(f"\n  ── Métricas — conjunto de prueba (20% cronológico) ──")
    print(f"     MAE  = {met['MAE']:.2f}  beneficiarios")
    print(f"     RMSE = {met['RMSE']:.2f}  beneficiarios")
    print(f"     R²   = {met['R2']:.4f}")

    ben_pred = predecir_beneficiarios(modelo, fecha_obj, df_despachos)
    print(f"\n  ✔ Beneficiarios estimados para {fecha_obj.date()} :  ŷ = {ben_pred}")

    # PASO 2: Módulo de Optimización
    _sep("PASO 2 — Módulo de Optimización del Kit (PLI — solver CBC)")
    res_opt = optimizar_kit(beneficiarios_estimados=ben_pred, df_inventario=df_inventario)

    print(f"  ✔ Estado del solver     : {res_opt['estado']}")
    print(f"  ✔ Puntaje obj. óptimo   : {res_opt['puntaje']}")
    print(f"\n  ── Kit óptimo seleccionado ({len(res_opt['kit'])} alimentos × 1 kg) ──")

    for alimento in res_opt["kit"]:
        s = df_inventario.loc[df_inventario["Categoria"] == alimento, "Stock_kg"]
        stock_str = f"{s.values[0]:>9.1f} kg" if len(s) else "       N/D"
        print(f"     • {alimento:<25}  Stock disponible: {stock_str}")

    print(f"\n  ── Cobertura nutricional del kit ──")
    print(f"     {'Nutriente':<18} {'Total':>10} {'Mínimo':>10} {'OK':>4}")
    print(f"     {'─'*46}")
    unidades = {"kcal": "kcal", "proteina": "g", "carbohidrato": "g", "grasa": "g", "fibra": "g"}
    for nut, vals in res_opt["nutricion"].items():
        u, ok = unidades[nut], "✔" if vals["cumple"] else "✘"
        print(f"     {nut:<18} {vals['total']:>8.1f}{u}  {vals['minimo']:>8}{u}  {ok:>3}")

    # PASO 3: Módulo de Sustitución Inteligente
    _sep("PASO 3 — Módulo de Sustitución Inteligente (distancia euclidiana normalizada)")
    sustituciones = activar_sustitucion(kit_propuesto=res_opt["kit"], beneficiarios=ben_pred, df_inventario=df_inventario)

    if not sustituciones:
        print("  ✔ Stock suficiente en todos los alimentos del kit. No se requiere sustitución.")
    else:
        print(f"  ⚠  Stock insuficiente detectado ({len(sustituciones)} alimento/s):\n")
        for original, sustituto in sustituciones.items():
            s = df_inventario.loc[df_inventario["Categoria"] == original, "Stock_kg"]
            st = f"{s.values[0]:.1f} kg" if len(s) else "N/D"
            if sustituto:
                df_sim  = calcular_similitud_nutricional(original, [sustituto])
                sim_val = df_sim.iloc[0]["Similitud"] if not df_sim.empty else 0.0
                print(f"     ✘ '{original}' (stock: {st})\n        → Sustituto: '{sustituto}'  (similitud: {sim_val:.4f})")
            else:
                print(f"     ✘ '{original}' (stock: {st}) → sin sustituto disponible")

    kit_final = list(res_opt["kit"])
    for original, sustituto in sustituciones.items():
        if sustituto and original in kit_final:
            kit_final[kit_final.index(original)] = sustituto

    # REPORTES FINALES DE CONTROL LOGÍSTICO
    _sep()
    print(f"  ✔  KIT FINAL — Jornada {fecha_obj.date()} | {ben_pred} beneficiarios estimados")
    print(f"  {'─'*75}")
    if not kit_final:
        print("     ⚠ NO SE PUDO GENERAR UN KIT VÁLIDO EN ESTA JORNADA.")
    else:
        print(f"     {'N°':<4} {'Categoría de Alimento':<30} {'Por Kit':<12} {'Total Jornada (kg)':<18}")
        print(f"     {'─'*71}")
        for i, alimento in enumerate(kit_final, 1):
            peso_por_kit = 1.0
            peso_total_jornada = peso_por_kit * ben_pred
            print(f"     {i:<4} {alimento:<30} {peso_por_kit:>7.1f} kg    {peso_total_jornada:>14,.1f} kg")
    _sep()

    return {
        "fecha_objetivo":        fecha_obj,
        "beneficiarios_pred":    ben_pred,
        "metricas_modelo":       met,
        "resultado_optimizador": res_opt,
        "sustituciones":         sustituciones,
        "kit_final":             kit_final,
        "df_despachos":          df_despachos,
        "df_inventario":         df_inventario,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fecha_arg = sys.argv[1] if len(sys.argv) > 1 else None
    resultados = ejecutar_pipeline(ruta_datos=DATA_PATH, fecha_objetivo=fecha_arg)