# Embedding visual for all sources with distinction between mentioned and not mentioned sources

# Replace the embeddings and metadata variables with your own data
embeddings = np.load('prompt_1_all_embeddings_simbad_test5_0_100.npy')
metadata = pd.read_csv('prompt_1_embeddings_metadata_0_100.csv')

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
embedding_2d = tsne.fit_transform(embeddings)

# Add t-SNE coordinates to metadata
metadata['text_emb1'] = embedding_2d[:, 0]
metadata['text_emb2'] = embedding_2d[:, 1]

# Save the metadata with coordinates for correlation analysis
metadata.to_csv('metadata_with_tsne_coords.csv', index=False)

# Map directly_mentioned column to marker styles
marker_map = {1: 'o', 0: 's'}  # circles for mentioned, squares for general summaries

plt.figure(figsize=(10, 8))

# Plot each group separately
for flag in [1, 0]:
    subset = metadata[metadata['source_flag'] == flag]
    plt.scatter(subset['text_emb1'], subset['text_emb2'],
                c=subset['hardness_ratio'],
                cmap='coolwarm', s=60, edgecolor='k',
                marker=marker_map[flag],
                label=f'Mentioned: {flag}')

plt.colorbar(label='Hardness Ratio')
plt.xlabel('text_emb1')
plt.ylabel('text_emb2')
plt.title('LLM Embeddings Colored by Hardness Ratio + Shape by Mentioned')
plt.legend()
plt.show()

print(f"Saved metadata with t-SNE coordinates to 'metadata_with_tsne_coords.csv'")
print(f"Columns: {list(metadata.columns)}")
