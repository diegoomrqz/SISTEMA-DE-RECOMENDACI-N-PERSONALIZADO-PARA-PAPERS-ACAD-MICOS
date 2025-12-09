# arxiv_personal_assistant.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Configuración de la página
st.set_page_config(
    page_title="ArXiv Research Assistant", 
    page_icon="🔬", 
    layout="wide"
)

# Definir rutas fijas
RUTA_BASE = r"C:\Diego\Visual studio\Redes Neuronales\Arxiv Seminario\Version 2\final"
RUTA_EMBEDDINGS = os.path.join(RUTA_BASE, "arxiv_embeddings_43453_20251129_221146.npy")
RUTA_METADATA = os.path.join(RUTA_BASE, "arxiv_embeddings_43453_20251129_221146_metadata.csv")

class ArXivPersonalAssistant:
    def __init__(self):
        self.embedding_model = None
        self.embeddings = None
        self.df = None
        self.data_loaded = False
        self.inicializar_modelo()
    
    def inicializar_modelo(self):
        """Inicializar el modelo de embeddings"""
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            st.sidebar.success("✅ Modelo de embeddings cargado")
            return True
        except Exception as e:
            st.error(f"❌ Error cargando modelo: {e}")
            return False
    
    def cargar_datos(self):
        """Cargar datos desde rutas fijas"""
        try:
            st.sidebar.info(f"📂 Rutas configuradas:")
            st.sidebar.info(f"- Embeddings: {os.path.basename(RUTA_EMBEDDINGS)}")
            st.sidebar.info(f"- Metadata: {os.path.basename(RUTA_METADATA)}")
            
            # Verificar que los archivos existan
            if not os.path.exists(RUTA_EMBEDDINGS):
                st.error(f"❌ No se encontró el archivo de embeddings: {RUTA_EMBEDDINGS}")
                return False
            
            if not os.path.exists(RUTA_METADATA):
                st.error(f"❌ No se encontró el archivo de metadata: {RUTA_METADATA}")
                return False
            
            # Cargar datos
            with st.spinner(f"Cargando embeddings desde {os.path.basename(RUTA_EMBEDDINGS)}..."):
                self.embeddings = np.load(RUTA_EMBEDDINGS)
            
            with st.spinner(f"Cargando metadata desde {os.path.basename(RUTA_METADATA)}..."):
                self.df = pd.read_csv(RUTA_METADATA)
            
            # Verificar dimensiones
            if len(self.df) != len(self.embeddings):
                st.warning(f"⚠️ Advertencia: El número de papers ({len(self.df)}) no coincide con el número de embeddings ({len(self.embeddings)})")
                # Tomar el mínimo para evitar errores
                min_len = min(len(self.df), len(self.embeddings))
                self.df = self.df.iloc[:min_len].copy()
                self.embeddings = self.embeddings[:min_len]
            
            # Verificar y crear columnas necesarias
            self._verificar_columnas()
            
            self.data_loaded = True
            st.success(f"✅ Sistema cargado con {len(self.df):,} papers")
            return True
            
        except Exception as e:
            st.error(f"❌ Error cargando datos: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def _verificar_columnas(self):
        """Verificar que existen todas las columnas necesarias"""
        columnas_necesarias = {
            'title': 'Título',
            'abstract': 'Resumen', 
            'category': 'Categoría',
            'authors': 'Autores',
            'published': 'Fecha publicación'
        }
        
        # Verificar columnas existentes
        columnas_existentes = []
        columnas_faltantes = []
        
        for col, desc in columnas_necesarias.items():
            if col not in self.df.columns:
                columnas_faltantes.append(desc)
            else:
                columnas_existentes.append(col)
        
        if columnas_faltantes:
            st.warning(f"⚠️ Columnas faltantes: {', '.join(columnas_faltantes)}")
        
        # Renombrar columnas si es necesario
        posibles_nombres = {
            'title': ['Title', 'title', 'paper_title', 'titulo'],
            'abstract': ['Abstract', 'abstract', 'summary', 'resumen'],
            'category': ['Category', 'category', 'categories', 'categoria'],
            'authors': ['Authors', 'authors', 'autores'],
            'published': ['Published', 'published', 'date', 'publication_date']
        }
        
        for col_nueva, posibles in posibles_nombres.items():
            if col_nueva not in self.df.columns:
                for posible in posibles:
                    if posible in self.df.columns and posible != col_nueva:
                        self.df[col_nueva] = self.df[posible]
                        break
        
        # Crear columnas de métricas si no existen
        metricas_default = {
            'quality_score': 0.5,
            'is_tutorial': 0,
            'is_application': 0, 
            'is_theoretical': 0,
            'is_recent': 0,
            'is_frontier': 0
        }
        
        for col, default_val in metricas_default.items():
            if col not in self.df.columns:
                self.df[col] = default_val
    
    def interfaz_principal(self):
        """Interfaz principal de la aplicación"""
        st.title("🔬 ArXiv Personal Research Assistant")
        st.markdown("### Encuentra los papers perfectos para tu investigación actual")
        
        # Sidebar con configuración del perfil
        perfil_usuario = self.sidebar_configuracion()
        
        # Área principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎯 Tu Búsqueda Personalizada")
            objetivo = st.text_area(
                "**Describe exactamente qué estás investigando:**",
                placeholder="Ej: 'Estoy trabajando en transformers para procesamiento de imágenes médicas usando attention mechanisms...'",
                height=100,
                key="objetivo_input"
            )
            
            if st.button("🚀 Buscar Papers Recomendados", type="primary", use_container_width=True):
                if objetivo:
                    if self.df is None or self.embeddings is None:
                        st.error("⚠️ Los datos no están cargados. Por favor, inicializa el sistema primero.")
                    else:
                        with st.spinner(f"🔍 Analizando {len(self.df):,} papers para encontrar los más relevantes..."):
                            recomendaciones = self.generar_recomendaciones(objetivo, perfil_usuario)
                            self.mostrar_resultados(recomendaciones, perfil_usuario)
                else:
                    st.warning("Por favor, describe tu investigación para obtener recomendaciones.")
        
        with col2:
            st.header("📊 Tu Perfil")
            self.mostrar_resumen_perfil(perfil_usuario)
    
    def sidebar_configuracion(self):
        """Sidebar con configuración del perfil de usuario"""
        with st.sidebar:
            st.header("👤 Configura tu Perfil")
            
            # 1. TIPO DE USUARIO
            st.subheader("1. Tu Rol")
            perfil = st.selectbox(
                "Selecciona tu perfil principal:",
                [
                    "🎓 Estudiante Pregrado", 
                    "🎓 Estudiante Maestría", 
                    "🎓 Estudiante Doctorado",
                    "🔬 Investigador Académico", 
                    "🏭 Investigador Industrial",
                    "👨‍🏫 Profesor/Educador",
                    "💼 Profesional Industria",
                    "🤖 Entusiasta/Aficionado"
                ]
            )
            
            # 2. ESPECIALIZACIÓN
            st.subheader("2. Tu Especialización")
            areas = st.multiselect(
                "Áreas de interés:",
                [
                    "Machine Learning", "Deep Learning", "Computer Vision", 
                    "Natural Language Processing", "Robotics", "Reinforcement Learning",
                    "Quantum Computing", "Physics", "Mathematics", "Statistics",
                    "Bioinformatics", "Computational Biology", "Neuroscience",
                    "Economics", "Finance", "Healthcare", "Medicine",
                    "Computer Systems", "Databases", "Software Engineering",
                    "Theory", "Algorithms", "Optimization"
                ],
                default=["Machine Learning", "Computer Vision"]
            )
            
            # 3. PREFERENCIAS DE CONTENIDO
            st.subheader("3. Preferencias de Contenido")
            
            st.write("**Nivel de profundidad:**")
            nivel = st.slider("", 1, 5, 3, 
                            help="1: Introductorio, 3: Balanceado, 5: Avanzado/Especializado")
            
            st.write("**Tipo de contenido preferido:**")
            col1, col2 = st.columns(2)
            with col1:
                tutorial = st.slider("🎓 Tutorial", 0.0, 1.0, 0.7)
                aplicado = st.slider("🏭 Aplicado", 0.0, 1.0, 0.8)
            with col2:
                teorico = st.slider("🔬 Teórico", 0.0, 1.0, 0.4)
                frontera = st.slider("🚀 Frontera", 0.0, 1.0, 0.6)
            
            # 4. ACTUALIDAD
            st.subheader("4. Preferencia Temporal")
            actualidad = st.slider("🆕 Papers recientes", 0.0, 1.0, 0.8,
                                 help="0: Cualquier fecha, 1: Solo últimos 2 años")
            
            return {
                'perfil': perfil,
                'areas': areas,
                'nivel': nivel,
                'tutorial': tutorial,
                'aplicado': aplicado, 
                'teorico': teorico,
                'frontera': frontera,
                'actualidad': actualidad
            }
    
    def mostrar_resumen_perfil(self, perfil):
        """Mostrar resumen visual del perfil"""
        # Radar chart de preferencias
        categorias = ['Tutorial', 'Aplicado', 'Teórico', 'Frontera', 'Actualidad']
        valores = [
            perfil['tutorial'], 
            perfil['aplicado'], 
            perfil['teorico'], 
            perfil['frontera'],
            perfil['actualidad']
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=valores,
            theta=categorias,
            fill='toself',
            fillcolor='rgba(100, 149, 237, 0.3)',
            line=dict(color='royalblue')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            height=300,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Información del perfil
        st.write(f"**Perfil:** {perfil['perfil']}")
        st.write(f"**Áreas:** {', '.join(perfil['areas'][:3])}{'...' if len(perfil['areas']) > 3 else ''}")
        st.write(f"**Nivel:** {perfil['nivel']}/5")
    
    def generar_recomendaciones(self, objetivo, perfil_usuario, n_resultados=15):
        """Generar recomendaciones personalizadas"""
        try:
            # Verificar que los datos estén cargados
            if self.df is None:
                st.error("Los datos no están cargados.")
                return []
            
            if self.embeddings is None:
                st.error("Los embeddings no están cargados.")
                return []
            
            # Embedding de la consulta
            consulta_embedding = self.embedding_model.encode([objetivo])
            
            # Calcular similitud semántica
            similitudes = cosine_similarity(consulta_embedding, self.embeddings)[0]
            
            # Calcular scores personalizados
            scores_personalizados = []
            
            for idx, paper in self.df.iterrows():
                score_base = similitudes[idx]
                
                # Ajustar según preferencias
                score_ajustado = score_base * 0.6  # 60% base semántica
                
                # Ajustar por tipo de contenido
                score_ajustado += paper.get('is_tutorial', 0) * 0.1 * perfil_usuario['tutorial']
                score_ajustado += paper.get('is_application', 0) * 0.1 * perfil_usuario['aplicado'] 
                score_ajustado += paper.get('is_theoretical', 0) * 0.1 * perfil_usuario['teorico']
                score_ajustado += paper.get('is_frontier', 0) * 0.1 * perfil_usuario['frontera']
                
                # Ajustar por actualidad
                if paper.get('is_recent', 0):
                    score_ajustado += 0.1 * perfil_usuario['actualidad']
                
                # Ajustar por calidad
                score_ajustado += paper.get('quality_score', 0.5) * 0.1
                
                scores_personalizados.append(score_ajustado)
            
            # Obtener top resultados
            scores_array = np.array(scores_personalizados)
            indices_top = np.argsort(scores_array)[::-1][:n_resultados]
            
            resultados = []
            for idx in indices_top:
                paper = self.df.iloc[idx]
                resultados.append({
                    'id': idx,
                    'titulo': paper['title'],
                    'categoria': paper['category'],
                    'score_total': scores_array[idx],
                    'score_semantico': similitudes[idx],
                    'abstract': paper['abstract'],
                    'autores': paper['authors'],
                    'publicado': paper.get('published', 'N/A'),
                    'es_tutorial': paper.get('is_tutorial', 0),
                    'es_aplicacion': paper.get('is_application', 0),
                    'es_teorico': paper.get('is_theoretical', 0),
                    'es_frontera': paper.get('is_frontier', 0),
                    'es_reciente': paper.get('is_recent', 0)
                })
            
            return resultados
            
        except Exception as e:
            st.error(f"Error generando recomendaciones: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return []
    
    def mostrar_resultados(self, recomendaciones, perfil_usuario):
        """Mostrar resultados de forma atractiva"""
        if not recomendaciones:
            st.warning("No se encontraron recomendaciones que coincidan con tu perfil.")
            return
        
        # Métricas generales
        st.header("📊 Resultados de tu Búsqueda")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_score = np.mean([r['score_total'] for r in recomendaciones])
            st.metric("🎯 Relevancia Promedio", f"{avg_score:.3f}")
        with col2:
            categorias = len(set([r['categoria'] for r in recomendaciones]))
            st.metric("🏷️ Categorías", categorias)
        with col3:
            recientes = sum([1 for r in recomendaciones if r['es_reciente']])
            st.metric("🆕 Recientes", recientes)
        with col4:
            st.metric("📚 Total", len(recomendaciones))
        
        # Mostrar cada recomendación
        st.header("📚 Papers Recomendados")
        
        for i, paper in enumerate(recomendaciones, 1):
            with st.container():
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    st.markdown(f"### {i}. {paper['titulo']}")
                    
                    # Información básica
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.write(f"**Categoría:** {paper['categoria']}")
                    with col_info2:
                        st.write(f"**Score:** {paper['score_total']:.3f}")
                    with col_info3:
                        st.write(f"**Publicado:** {paper['publicado']}")
                    
                    # Indicadores
                    indicadores = []
                    if paper['es_tutorial']:
                        indicadores.append("🎓 Tutorial")
                    if paper['es_aplicacion']:
                        indicadores.append("🏭 Aplicación") 
                    if paper['es_teorico']:
                        indicadores.append("🔬 Teórico")
                    if paper['es_frontera']:
                        indicadores.append("🚀 Frontera")
                    if paper['es_reciente']:
                        indicadores.append("🆕 Reciente")
                    
                    if indicadores:
                        st.write(" | ".join(indicadores))
                    
                    # Abstract con expander
                    with st.expander("📝 Ver Abstract"):
                        st.write(paper['abstract'])
                        st.write(f"**Autores:** {paper['autores']}")
                        st.write(f"**ID:** {paper['id']}")
                
                with col_right:
                    # Gráfico de score
                    scores = {
                        'Semántico': paper['score_semantico'],
                        'Total': paper['score_total']
                    }
                    
                    fig = go.Figure(go.Bar(
                        x=list(scores.values()),
                        y=list(scores.keys()),
                        orientation='h',
                        marker_color=['lightblue', 'royalblue']
                    ))
                    
                    fig.update_layout(
                        height=150,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(range=[0, 1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
        
        # Opción para descargar resultados
        df_resultados = pd.DataFrame(recomendaciones)
        csv = df_resultados.to_csv(index=False)
        
        st.download_button(
            label="💾 Descargar Recomendaciones (CSV)",
            data=csv,
            file_name=f"arxiv_recomendaciones_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

def main():
    """Función principal"""
    st.sidebar.title("🔬 ArXiv Assistant")
    
    # Mostrar información de rutas
    st.sidebar.write(f"**Ruta base:**")
    st.sidebar.code(RUTA_BASE)
    st.sidebar.write(f"**Archivo embeddings:**")
    st.sidebar.code(os.path.basename(RUTA_EMBEDDINGS))
    st.sidebar.write(f"**Archivo metadata:**")
    st.sidebar.code(os.path.basename(RUTA_METADATA))
    
    # Verificar instalación
    st.sidebar.write("### Verificación del Sistema")
    
    assistant = ArXivPersonalAssistant()
    
    if st.sidebar.button("🔄 Inicializar Sistema", use_container_width=True):
        with st.spinner("Cargando datos..."):
            if assistant.cargar_datos():
                st.session_state['assistant'] = assistant
                st.sidebar.success("✅ Sistema listo")
            else:
                st.sidebar.error("❌ Error cargando datos")
    
    # Si ya se cargaron datos, mostrar la interfaz principal
    if 'assistant' in st.session_state:
        assistant = st.session_state['assistant']
        assistant.interfaz_principal()
    else:
        # Mostrar instrucciones iniciales
        st.info("👈 Por favor, haz clic en 'Inicializar Sistema' para comenzar.")
        st.write("### 📋 Instrucciones:")
        st.write("1. Haz clic en 'Inicializar Sistema' en la barra lateral")
        st.write("2. Configura tu perfil en la barra lateral")
        st.write("3. Describe tu investigación en el área principal")
        st.write("4. Haz clic en 'Buscar Papers Recomendados'")
    
    # Información del sistema
    st.sidebar.write("---")
    st.sidebar.write("### 📋 Instrucciones:")
    st.sidebar.write("1. Click en 'Inicializar Sistema'")
    st.sidebar.write("2. Configura tu perfil a la izquierda") 
    st.sidebar.write("3. Describe tu investigación")
    st.sidebar.write("4. Click en 'Buscar Papers'")

if __name__ == "__main__":
    main()