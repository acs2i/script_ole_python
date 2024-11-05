import pandas as pd
import pymongo
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
import os
from datetime import datetime, timezone
import numpy as np
from tqdm import tqdm
import logging
import sys
from typing import Dict, List, Any, Tuple
import gc
from collections import defaultdict
import csv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('import.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configurations d'import
IMPORT_CONFIGS = {
    'COL': {
        'collection': 'collection',
        'file_path': 'RDD_PREREF-01_COL_20241001_121940.csv',
        'date_fields': ['creationdate', 'modificationdate'],
        'field_mappings': {
            'creationdate': 'creation_date',
            'modificationdate': 'modification_date'
        },
        'default_values': {
            'creator_id': ''
        },
        'indexes': [
            ([('code', 1)], {'unique': True})
        ]
    },
    'BRAND': {
        'collection': 'brand',
        'file_path': 'RDD_PREREF-02_MRQ_20241001_122715.csv',
        'date_fields': ['creationdate', 'modificationdate'],
        'field_mappings': {
            'creationdate': 'creation_date',
            'modificationdate': 'modification_date'
        },
        'default_values': {
            'creator_id': ''
        },
        'indexes': [
            ([('code', 1)], {'unique': True})
        ]
    },
    'CLASS': {
        'collection': 'tag',
        'file_path': 'RDD_PREREF-03_CLASS_20241001_123605.csv',
        'date_fields': ['creationdate', 'modificationdate'],
        'field_mappings': {
            'creationdate': 'creation_date',
            'modificationdate': 'modification_date'
        },
        'default_values': {
            'creator_id': '',
            'tag_grouping_id': ObjectId('6698d83baf0264575e513a5d')
        },
        'indexes': [
            ([('code', 1), ('level', 1)], {'unique': True})
        ]
    },
    'DIMENSION': {
        'collection': 'dimension',
        'file_path': 'RDD_PREREF-04_DIM_20241001_124242.csv',
        'date_fields': ['creationdate', 'modificationdate'],
        'field_mappings': {
            'creationdate': 'creation_date',
            'modificationdate': 'modification_date'
        },
        'default_values': {
            'creator_id': ''
        },
        'indexes': [
            ([('type', 1), ('code', 1)], {'unique': True})
        ],
        'requires_preprocessing': True
    },
    'DIMGRIL': {
        'collection': 'dimension_grid',
        'file_path': 'RDD_PREREF-05_DIMGRIL_20241001_125455.csv',
        'indexes': [
            ([('code', 1)], {'unique': True})
        ],
        'requires_preprocessing': True
    },
    'SUPPLIER': {
        'collection': 'supplier',
        'file_path': 'RDD_PREREF-06_FOURN_20241001_135437.csv',
        'date_fields': ['creationdate', 'modificationdate'],
        'field_mappings': {
            'creationdate': 'creation_date',
            'modificationdate': 'modification_date',
            'customerref': 'customer_ref',
            'company_name': 'company_name',
            'web_url': 'web_url',
            'trade_name': 'trade_name',
            'address1': 'address_1',
            'address2': 'address_2',
            'address3': 'address_3'
        },
        'default_values': {
            'additional_fields': [],
            'brand_id': [],
            'contacts': []
        },
        'indexes': [
            ([('code', 1)], {'unique': True})
        ]
    },
    'PRODUCT': {
        'collection': 'product',
        'file_path': 'RDD_PREREF-07_ART_20241010_161413.csv',
        'indexes': [
            ([('reference', 1)], {'unique': True})
        ],
        'requires_reference_data': True
    },
    'UVC': {
        'collection': 'uvc',
        'file_path': 'RDD_PREREF-08_UVC_20241010_162610.csv',
        'indexes': [
            ([('reference', 1), ('dimensions', 1)], {'unique': True})
        ],
        'requires_reference_data': True
    }
}

class MongoImporter:
    def __init__(self, mongo_url: str, db_name: str, chunk_size: int = 5000):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.chunk_size = chunk_size
        self.client = None
        self.db = None
        self.reference_data = {}
        self.logger = logging.getLogger(__name__)

    def connect(self):
        """Établit la connexion MongoDB"""
        self.client = MongoClient(self.mongo_url)
        self.db = self.client[self.db_name]
        self.logger.info("Connected to MongoDB")

    def disconnect(self):
        """Ferme la connexion MongoDB"""
        if self.client:
            self.client.close()
            self.logger.info("Disconnected from MongoDB")

    def read_csv_safely(self, file_path: str, chunk_size: int = None) -> pd.DataFrame:
        """Lit un fichier CSV de manière sécurisée avec gestion d'encodage"""
        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                self.logger.info(f"Trying to read with {encoding} encoding...")
                
                # Déterminer le nombre de colonnes
                with open(file_path, 'r', encoding=encoding) as f:
                    header = f.readline().strip()
                    num_columns = len(header.split(';'))
                    self.logger.info(f"Detected {num_columns} columns in header")
    
                # Options de lecture CSV
                csv_options = {
                    'sep': ';',
                    'dtype': str,
                    'keep_default_na': False,
                    'on_bad_lines': 'skip',  # Skip les lignes problématiques
                    'quoting': csv.QUOTE_MINIMAL,
                    'encoding': encoding,
                    'encoding_errors': 'replace',  # Remplacer les caractères illisibles
                    'escapechar': '\\',
                    'low_memory': False
                }
    
                if chunk_size:
                    csv_options['chunksize'] = chunk_size
    
                return pd.read_csv(file_path, **csv_options)
    
            except UnicodeDecodeError:
                self.logger.warning(f"Failed to read with {encoding} encoding, trying next...")
                continue
            except Exception as e:
                if "charmap" in str(e):
                    continue
                self.logger.error(f"Error reading CSV file {file_path}: {str(e)}")
                raise
    
        raise ValueError(f"Unable to read {file_path} with any of the attempted encodings")

    def load_reference_data(self):
        """Charge les données de référence"""
        self.logger.info("Loading reference data...")
        self.reference_data = {
            'tags': {doc['code']: doc['_id'] for doc in self.db.tag.find({}, {'code': 1})},
            'brands': {doc['code']: doc['_id'] for doc in self.db.brand.find({}, {'code': 1})},
            'collections': {doc['code']: doc['_id'] for doc in self.db.collection.find({}, {'code': 1})},
            'suppliers': {doc['code']: doc['_id'] for doc in self.db.supplier.find({}, {'code': 1})}
        }
        self.logger.info("Reference data loaded")

    def preprocess_dimgril(self, df: pd.DataFrame) -> List[Dict]:
        """Prétraitement des données DIMGRIL"""
        grouped = defaultdict(lambda: {'dimensions': [], 'frn_labels': []})
        for _, row in df.iterrows():
            grouped[row['code']]['dimensions'].append(row['avccodedim'])
            grouped[row['code']]['frn_labels'].append(row['frnlabel'])
        
        return [
            {
                'code': code,
                'dimensions': data['dimensions'],
                'frn_labels': data['frn_labels'],
                'status': 'A',
                'type': 'taille'
            }
            for code, data in grouped.items()
        ]

    def preprocess_dimension(self, df: pd.DataFrame) -> List[Dict]:
        """Prétraitement des données DIMENSION"""
        return [
            {
                'type': row['type'].lower(),
                'code': row['code'],
                'label': row['label'],
                'status': row['status'],
                'creation_date': pd.to_datetime(row['creationdate']) if pd.notna(row['creationdate']) else None,
                'modification_date': pd.to_datetime(row['modificationdate']) if pd.notna(row['modificationdate']) else None,
                'creator_id': ''
            }
            for _, row in df.iterrows()
        ]

    def process_basic_import(self, config_name: str):
        """Traitement des imports de base"""
        config = IMPORT_CONFIGS[config_name]
        self.logger.info(f"Processing {config_name}...")
        
        collection = self.db[config['collection']]
        collection.delete_many({})

        try:
            reader = self.read_csv_safely(config['file_path'], self.chunk_size)
            chunks = reader if isinstance(reader, pd.io.parsers.TextFileReader) else [reader]

            total_processed = 0
            skipped_rows = 0

            for chunk in chunks:
                try:
                    if config.get('requires_preprocessing'):
                        if config_name == 'DIMGRIL':
                            processed_data = self.preprocess_dimgril(chunk)
                        elif config_name == 'DIMENSION':
                            processed_data = self.preprocess_dimension(chunk)
                    else:
                        processed_data = []
                        for _, row in chunk.iterrows():
                            try:
                                doc = {**config.get('default_values', {})}
                                
                                for col in row.index:
                                    field_name = config.get('field_mappings', {}).get(col, col)
                                    value = row[col]
                                    
                                    if field_name in config.get('date_fields', []):
                                        try:
                                            value = pd.to_datetime(value) if pd.notna(value) else None
                                        except:
                                            value = None
                                    
                                    doc[field_name] = value
                                
                                processed_data.append(doc)
                            except Exception as row_error:
                                skipped_rows += 1
                                self.logger.warning(f"Skipped row in {config_name}: {str(row_error)}")
                                continue

                    if processed_data:
                        collection.insert_many(processed_data, ordered=False)
                        total_processed += len(processed_data)
                        self.logger.info(f"Processed {total_processed} records for {config_name} (Skipped: {skipped_rows})")

                except Exception as chunk_error:
                    self.logger.error(f"Error processing chunk in {config_name}: {str(chunk_error)}")
                    continue

            for index_fields, index_options in config.get('indexes', []):
                collection.create_index(index_fields, **index_options)

            self.logger.info(f"{config_name} completed: {total_processed} processed, {skipped_rows} skipped")

        except Exception as e:
            self.logger.error(f"Failed to process {config_name}: {str(e)}")
            raise
            
    @staticmethod
    def safe_convert(value, default=0.0):
        """Convertit une valeur en integer de manière sécurisée"""
        if value is None or value == '':
            return default
        try:
            # Supprimer les espaces et remplacer la virgule par un point
            cleaned_value = str(value).strip().replace(',', '.')
            # Convertir en float d'abord pour gérer les nombres décimaux
            float_value = float(cleaned_value)
            # Retourner la partie entière
            return float_value
        except (ValueError, TypeError):
            return default

    def process_product(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Traitement des données produit"""
        products = []
        for _, row in df.iterrows():
            tag_ids = [
                self.reference_data['tags'][code]
                for code in [row.get('famcode'), row.get('sfamcode'), row.get('ssfamcode')]
                if code and code in self.reference_data['tags']
            ]

            brand_id = self.reference_data['brands'].get(row.get('brandcode'))
            collection_id = self.reference_data['collections'].get(row.get('collectioncode'))
            supplier_id = self.reference_data['suppliers'].get(row.get('supplier'))

            product = {
                'creator_id': '',
                'reference': row.get('reference', ''),
                'alias': row.get('alias', ''),
                'short_label': row.get('short_label', ''),
                'long_label': row.get('long_label', ''),
                'status': row.get('status', 'I'),
                'type': row.get('type', 'Marchandise'),
                'brand_ids': [brand_id] if brand_id else [],
                'dimension_type_id': '',
                'dimension_ids': [],
                'tag_ids': tag_ids,
                'weight_measure_unit': 'KG',
                'net_weight': MongoImporter.safe_convert(row.get('net_weight')),
                'gross_weight': MongoImporter.safe_convert(row.get('gross_weight')),
                'dimension_measure_unit': 'M',
                'height': MongoImporter.safe_convert(row.get('height')),
                'length': MongoImporter.safe_convert(row.get('length')),
                'width': MongoImporter.safe_convert(row.get('width')),
                'taxcode': row.get('taxcode', ''),
                'paeu': MongoImporter.safe_convert(row.get('paeu')),
                'tbeu_pb': MongoImporter.safe_convert(row.get('tbeu_pb')),
                'tbeu_pmeu': MongoImporter.safe_convert(row.get('tbeu_pmeu')),
                'comment': row.get('comment', ''),
                'blocked': row.get('blocked', 'Non'),
                'collection_ids': [collection_id] if collection_id else [],
                'suppliers': [{
                    'supplier_id': supplier_id,
                    'supplier_ref': row.get('supplierref', ''),
                    'pcb': row.get('pcb', '1'),
                    'made_in': row.get('madein', ''),
                    'custom_cat': row.get('custom_category', '')
                }] if supplier_id else [],
                'tag_grouping_ids': [],
                'uvc_ids': [],
                'updatedAt': datetime.now(timezone.utc)
            }
            products.append(product)

        return products

    def process_uvc(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Traitement des données UVC"""
        product_refs = {doc['reference']: doc['_id'] 
                       for doc in self.db.product.find({}, {'reference': 1})}
        
        uvcs = []
        for _, row in df.iterrows():
            if row.get('reference') not in product_refs:
                continue

            uvc = {
                'dimensions': f"{row.get('colorcode', '')}/{row.get('sizecode', '')}",
                'eans': [x for x in [
                    f"{row.get('reference')}_{row.get('colorcode')}_{row.get('sizecode')}",
                    row.get('colombusskucode'),
                    row.get('ean13_p')
                ] if x],
                'ean': '',
                'supplier_ref': row.get('supplier', ''),
                'supplier_ref_code': row.get('supplierref', ''),
                'madein': row.get('madein', ''),
                'collection_code': row.get('collectioncode', ''),
                'custom_category': row.get('customcategory', ''),
                'reference': row.get('reference', ''),
                'net_weight': MongoImporter.safe_convert(row.get('netweight')),
                'gross_weight': MongoImporter.safe_convert(row.get('grossweight')),
                'height': MongoImporter.safe_convert(row.get('height')),
                'length': MongoImporter.safe_convert(row.get('length')),
                'width': MongoImporter.safe_convert(row.get('width')),
                'prices': [{
                    'paeu': MongoImporter.safe_convert(row.get('paeu')),
                    'tbeu_pb': row.get('tbeu_pb', ''),
                    'tbeu_pmeu': row.get('tbeu_pmeu', '')
                }],
                'blocked': row.get('blocked', 'Non'),
                'blocked_reason_code': row.get('blockedreasoncode', ''),
                'coulfour': row.get('01_coulfour', ''),
                'status': row.get('status', 'I'),
                'visible_on_internet': row.get('03_visiblesurinternet', ''),
                'sold_on_internet': row.get('04_ventesurinternet', ''),
                'seuil_internet': row.get('05_seuilinternet', ''),
                'en_reassort': row.get('06_enreassort', ''),
                'remisegenerale': row.get('07_remisegenerale', ''),
                'fixation': row.get('08_fixation', ''),
                'ventemetre': row.get('09_ventemetre', ''),
                'comment': row.get('10_commentaire', ''),
                'product_id': product_refs[row.get('reference')]
            }
            uvcs.append(uvc)

        return uvcs

    def process_csv_in_chunks(self, config_name: str) -> Tuple[int, int]:
        """Traitement des fichiers CSV par morceaux avec gestion d'erreurs"""
        config = IMPORT_CONFIGS[config_name]
        file_path = config['file_path']
        
        try:
            # Initialisation
            collection = self.db[config['collection']]
            collection.delete_many({})
    
            # Création des index
            for index_fields, index_options in config.get('indexes', []):
                collection.create_index(index_fields, **index_options)
    
            # Lecture du CSV
            total_processed = 0
            skipped_rows = 0
            
            # Estimation du nombre total de lignes
            with open(file_path, 'rb') as f:
                total_lines = sum(1 for _ in f)
            total_chunks = (total_lines - 1) // self.chunk_size + 1  # -1 pour l'en-tête
    
            chunks = self.read_csv_safely(file_path, self.chunk_size)
            
            with tqdm(total=total_chunks, desc=f"Processing {config_name}") as pbar:
                for chunk_index, chunk in enumerate(chunks):
                    try:
                        # Nettoyage des données
                        chunk = chunk.replace({pd.NA: '', None: ''})
                        chunk = chunk.fillna('')
                        
                        # Traitement selon le type
                        if config_name == 'PRODUCT':
                            processed_data = self.process_product(chunk)
                        elif config_name == 'UVC':
                            processed_data = self.process_uvc(chunk)
                        else:
                            continue
    
                        if processed_data:
                            # Insertion par lots
                            result = collection.insert_many(
                                processed_data,
                                ordered=False,
                                bypass_document_validation=True
                            )
                            inserted_count = len(result.inserted_ids)
                            total_processed += inserted_count
                            skipped_rows += len(chunk) - inserted_count
    
                        # Nettoyage mémoire
                        del processed_data
                        gc.collect()
    
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
                        skipped_rows += len(chunk)
                        continue
                    finally:
                        pbar.update(1)
                        pbar.set_postfix({
                            'processed': total_processed,
                            'skipped': skipped_rows
                        })
    
            return total_processed, skipped_rows

        except Exception as e:
            self.logger.error(f"Failed to process {config_name}: {str(e)}")
            raise

    def update_product_references(self):
        """Mise à jour des références UVC dans les produits"""
        self.logger.info("Updating product references...")
        
        try:
            # Regroupement des UVCs par produit
            uvc_by_product = defaultdict(list)
            cursor = self.db.uvc.find({}, {'product_id': 1})
            
            for uvc in cursor:
                if 'product_id' in uvc:
                    uvc_by_product[str(uvc['product_id'])].append(uvc['_id'])

            # Mise à jour par lots
            total_updates = len(uvc_by_product)
            processed = 0
            bulk_ops = []
            
            with tqdm(total=total_updates, desc="Updating products") as pbar:
                for product_id, uvc_ids in uvc_by_product.items():
                    bulk_ops.append(UpdateOne(
                        {'_id': ObjectId(product_id)},
                        {'$set': {'uvc_ids': uvc_ids}},
                        upsert=False
                    ))

                    if len(bulk_ops) >= self.chunk_size:
                        try:
                            result = self.db.product.bulk_write(bulk_ops, ordered=False)
                            processed += len(bulk_ops)
                            bulk_ops = []
                            pbar.update(self.chunk_size)
                        except Exception as e:
                            self.logger.error(f"Bulk update error: {str(e)}")
                            bulk_ops = []

                # Traitement final
                if bulk_ops:
                    try:
                        result = self.db.product.bulk_write(bulk_ops, ordered=False)
                        processed += len(bulk_ops)
                        pbar.update(len(bulk_ops))
                    except Exception as e:
                        self.logger.error(f"Final bulk update error: {str(e)}")

            self.logger.info(f"Updated {processed}/{total_updates} products")

        except Exception as e:
            self.logger.error(f"Failed to update product references: {str(e)}")
            raise

    def import_all(self):
        """Exécution de l'import complet"""
        try:
            # Connexion
            self.connect()
            
            # Import des données de référence
            basic_configs = ['COL', 'BRAND', 'CLASS', 'DIMENSION', 'DIMGRIL', 'SUPPLIER']
            for config_name in basic_configs:
                self.logger.info(f"Starting import of {config_name}")
                self.process_basic_import(config_name)
            
            # Chargement des références
            self.load_reference_data()
            
            # Import des produits
            self.logger.info("Starting product import")
            product_processed, product_skipped = self.process_csv_in_chunks('PRODUCT')
            self.logger.info(f"Products: {product_processed} processed, {product_skipped} skipped")
            
            # Import des UVCs
            self.logger.info("Starting UVC import")
            uvc_processed, uvc_skipped = self.process_csv_in_chunks('UVC')
            self.logger.info(f"UVCs: {uvc_processed} processed, {uvc_skipped} skipped")
            
            # Mise à jour des références
            self.update_product_references()
            
            self.logger.info("Import completed successfully")
            
        except Exception as e:
            self.logger.error(f"Import failed: {str(e)}")
            raise
        finally:
            self.disconnect()

def main():
    """Point d'entrée principal"""
    import_params = {
        'mongo_url': 'mongodb://192.168.10.235:27017',
        'db_name': 'ole_test',
        'chunk_size': 5000
    }

    try:
        importer = MongoImporter(**import_params)
        importer.import_all()
    except Exception as e:
        logging.error(f"Failed to run import: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
