
## JSON Document of 36K PRIDE Studies
Single JSON document with list of PRIDE study documents: 
`https://edwardslab.bmcb.georgetown.edu/~nedwards/dropbox/6ItUS2tEdC/pride-studies.json`.

## Individual JSON Documents for each Study
Use url `https://edwardslab.bmcb.georgetown.edu/~nedwards/dropbox/6ItUS2tEdC/PRIDE/<accession>.json` for single JSON document. PRIDE accessions look like: PXD018757.

## CouchDB Database
A CouchDB JSON document store is available for read-only access to the PRIDE study JSON documents. Use URL `https://edwardslab.bmcb.georgetown.edu/pride-couchdb` for the CouchDB server to access database `pride`. Read-only access available for user `public` with password `public`. See [`couchdb`](../couchdb) for this resource.

## OpenAI test-embeddings
Text embeddings for a markdown document formed from the descriptions in the PRIDE study are available, using OpenAI's v3 text-embeddings - small and large. A CSV file provides the text that was embedded and any additional metadata, while a Feather-format file provides the actual embeddings. See [Embeddings.ipynb](Embeddings.ipynb).

