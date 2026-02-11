import streamlit as st
import os
import plotly.express as px
import pandas as pd
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Analyse des Images Organisées par Âge",
    layout="wide"
)

# Titre de l'application
st.title("📊 Analyse des Images Organisées par Tranche d'Âge")

# Chemin vers les données organisées
data_path = r"data\images_organisees"

# Fonction pour compter les images dans le dossier organisé
def count_organized_images(path):
    counts = {}
    total_images = 0
    
    # Tranches d'âge attendues
    age_groups = ["1-20", "21-50", "51-100"]
    
    for age_group in age_groups:
        age_path = os.path.join(path, age_group)
        if os.path.exists(age_path):
            # Compter les fichiers images
            images = [f for f in os.listdir(age_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            count = len(images)
            counts[age_group] = count
            total_images += count
        else:
            counts[age_group] = 0
    
    return counts, total_images

# Fonction pour obtenir les statistiques détaillées
def get_detailed_stats(path):
    stats = []
    
    for age_group in ["1-20", "21-50", "51-100"]:
        age_path = os.path.join(path, age_group)
        
        if os.path.exists(age_path):
            # Lister toutes les images
            images = [f for f in os.listdir(age_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            # Calculer la taille totale
            total_size_mb = 0
            image_details = []
            
            for img in images:
                img_path = os.path.join(age_path, img)
                size_bytes = os.path.getsize(img_path)
                size_mb = size_bytes / (1024 * 1024)
                total_size_mb += size_mb
                
                # Vérifier si l'image commence par "Screenshot"
                if img.startswith('Screenshot'):
                    # Pour les Screenshots, on met simplement la tranche d'âge
                    age = f"({age_group})"
                else:
                    # Sinon, extraire l'âge du nom de fichier normalement
                    age = None
                    name_without_ext = os.path.splitext(img)[0]
                    if '_MALE_' in name_without_ext:
                        try:
                            age_part = name_without_ext.split('_MALE_')[-1]
                            age = int(age_part.split('_')[0])
                        except:
                            pass
                    elif '_FEMALE_' in name_without_ext:
                        try:
                            age_part = name_without_ext.split('_FEMALE_')[-1]
                            age = int(age_part.split('_')[0])
                        except:
                            pass
                    else:
                        try:
                            age_part = name_without_ext.split('_NONE_')[-1]
                            age = int(age_part.split('_')[0])
                        except:
                            pass   
                
                image_details.append({
                    'Nom': img,
                    'Taille_MB': round(size_mb, 2),
                    'Âge': age
                })
            
            # Calculer les statistiques
            avg_size = total_size_mb / len(images) if images else 0
            
            stats.append({
                'Tranche d\'âge': age_group,
                'Nombre d\'images': len(images),
                'Taille totale (MB)': round(total_size_mb, 2),
                'Taille moyenne (MB)': round(avg_size, 2),
                'Détails images': image_details
            })
        else:
            stats.append({
                'Tranche d\'âge': age_group,
                'Nombre d\'images': 0,
                'Taille totale (MB)': 0,
                'Taille moyenne (MB)': 0,
                'Détails images': []
            })
    
    return stats

# Vérification de l'existence du dossier
if os.path.exists(data_path):
    # Section 1: Métriques principales
    st.header("📈 Métriques Générales")
    
    counts, total_images = count_organized_images(data_path)
    
    # Afficher les métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", total_images)
    
    # Calculer les pourcentages pour chaque tranche
    for i, (age_group, count) in enumerate(counts.items()):
        with [col2, col3, col4][i]:
            percentage = (count / total_images * 100) if total_images > 0 else 0
            st.metric(
                f"{age_group} ans",
                count,
                delta=f"{percentage:.1f}% du total"
            )
    
    # Section 2: Graphiques
    st.header("📊 Visualisations")
    
    # Préparer les données pour les graphiques
    df_counts = pd.DataFrame([
        {"Tranche d'âge": age_group, "Nombre d'images": count}
        for age_group, count in counts.items()
    ])
    
    # Graphique à barres
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            df_counts,
            x="Tranche d'âge",
            y="Nombre d'images",
            title="Nombre d'images par tranche d'âge",
            color="Tranche d'âge",
            text="Nombre d'images"
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Diagramme en camembert
    with col2:
        fig_pie = px.pie(
            df_counts,
            names="Tranche d'âge",
            values="Nombre d'images",
            title="Répartition par tranche d'âge",
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Section 3: Statistiques détaillées
    st.header("📋 Statistiques Détaillées")
    
    # Obtenir les statistiques détaillées
    detailed_stats = get_detailed_stats(data_path)
    df_detailed = pd.DataFrame(detailed_stats)
    
    # Afficher le tableau des statistiques
    st.dataframe(
        df_detailed[['Tranche d\'âge', 'Nombre d\'images', 
                     'Taille totale (MB)', 'Taille moyenne (MB)']],
        use_container_width=True
    )
    
    # Section 4: Détails par tranche d'âge (expandable)
    st.header("🔍 Détails des Images par Tranche")
    
    for stat in detailed_stats:
        if stat['Nombre d\'images'] > 0:
            with st.expander(f"{stat['Tranche d\'âge']} - {stat['Nombre d\'images']} images"):
                # Créer un DataFrame pour les images de cette tranche
                details_df = pd.DataFrame(stat['Détails images'])
                
                # Afficher le tableau
                st.dataframe(
                    details_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Statistiques sur les âges extraits (uniquement les âges numériques)
                if details_df['Âge'].notna().any():
                    # Filtrer uniquement les âges numériques pour les statistiques
                    numeric_ages = []
                    screenshot_count = 0
                    
                    for value in details_df['Âge'].dropna():
                        if isinstance(value, str) and value.startswith('Screenshot'):
                            screenshot_count += 1
                        elif isinstance(value, (int, float)):
                            numeric_ages.append(value)
                        elif isinstance(value, str) and value.isdigit():
                            try:
                                numeric_ages.append(int(value))
                            except:
                                pass
                    
                    # Afficher les statistiques des screenshots
                    if screenshot_count > 0:
                        st.info(f"📸 **{screenshot_count} image(s) Screenshot** (classées dans {stat['Tranche d\'âge']})")
                    
                    # Afficher les statistiques des âges numériques
                    if numeric_ages:
                        ages_series = pd.Series(numeric_ages)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Âge moyen", f"{ages_series.mean():.1f} ans")
                        with col2:
                            st.metric("Âge min", f"{ages_series.min():.0f} ans")
                        with col3:
                            st.metric("Âge max", f"{ages_series.max():.0f} ans")
                
                # Graphique de répartition des tailles
                if not details_df.empty:
                    fig_size_dist = px.histogram(
                        details_df,
                        x="Taille_MB",
                        title=f"Distribution des tailles - {stat['Tranche d\'âge']}",
                        nbins=20
                    )
                    fig_size_dist.update_layout(
                        xaxis_title="Taille (MB)",
                        yaxis_title="Nombre d'images"
                    )
                    st.plotly_chart(fig_size_dist, use_container_width=True)
    
    # Section 5: Informations techniques
    st.header("ℹ️ Informations Techniques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Chemin analysé:** {os.path.abspath(data_path)}
        
        **Structure des dossiers:**
        ```
        {data_path}/
        ├── 1-20/
        ├── 21-50/
        └── 51-100/
        ```
        
        **Détection des Screenshots:**
        - Les images commençant par "Screenshot" affichent leur tranche d'âge
        - Exemple: "Screenshot (1-20)" pour un screenshot dans le dossier 1-20
        """)
    
    with col2:
        # Vérifier l'existence des dossiers
        folders_exist = []
        for age_group in ["1-20", "21-50", "51-100"]:
            folder_path = os.path.join(data_path, age_group)
            exists = os.path.exists(folder_path)
            folders_exist.append({
                "Dossier": age_group,
                "Existe": "✅" if exists else "❌",
                "Images": counts[age_group]
            })
        
        st.table(pd.DataFrame(folders_exist))
    
    # Section 6: Export des données
    st.header("💾 Export des Données")
    
    if st.button("📥 Exporter les statistiques en CSV"):
        # Préparer les données pour l'export
        export_data = []
        for stat in detailed_stats:
            for detail in stat['Détails images']:
                export_data.append({
                    'Tranche_d_age': stat['Tranche d\'âge'],
                    'Nom_fichier': detail['Nom'],
                    'Taille_MB': detail['Taille_MB'],
                    'Âge_extraite': detail['Âge']
                })
        
        export_df = pd.DataFrame(export_data)
        
        # Afficher un aperçu
        st.write("Aperçu des données à exporter:")
        st.dataframe(export_df.head(10), use_container_width=True)
        
        # Convertir en CSV
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        # Bouton de téléchargement
        st.download_button(
            label="Télécharger le fichier CSV",
            data=csv_data,
            file_name="statistiques_images_organisees.csv",
            mime="text/csv"
        )

else:
    st.error(f"⚠️ Le dossier organisé n'existe pas : {data_path}")
    
    st.info("""
    **Pour résoudre ce problème:**
    
    1. Vérifiez que le script d'organisation a bien été exécuté
    2. Vérifiez le chemin du dossier: `data/images_organisees`
    3. Si nécessaire, exécutez d'abord le script d'organisation des images
    """)
    
    # Option pour créer la structure si elle n'existe pas
    if st.button("Créer la structure de dossiers vide"):
        for age_group in ["1-20", "21-50", "51-100"]:
            os.makedirs(os.path.join("data", "images_organisees", age_group), exist_ok=True)
        st.success("Structure de dossiers créée!")
        st.rerun()