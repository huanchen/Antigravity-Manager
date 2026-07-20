/**
 * 模型分类工具函数（无 React / icons 依赖，可在 Node 环境直接导入）
 */

export type ModelCategory =
    | 'gemini-pro'
    | 'gemini-flash'
    | 'gemini-pro-image'
    | 'gemini-flash-image'
    | 'claude-sonnet'
    | 'claude-opus'
    | 'claude'
    | 'other';

/** Canonical quota buckets exposed by the upstream Claude account API. */
export const CLAUDE_SONNET_QUOTA_MODEL = 'claude-sonnet-4-6' as const;
export const CLAUDE_OPUS_QUOTA_MODEL = 'claude-opus-4-6-thinking' as const;
export const LEGACY_CLAUDE_QUOTA_MODEL = 'claude' as const;
export const SUPPORTED_CLAUDE_QUOTA_MODELS = [
    CLAUDE_SONNET_QUOTA_MODEL,
    CLAUDE_OPUS_QUOTA_MODEL,
] as const;

export function categorizeModel(name: string): ModelCategory {
    const n = name.trim().toLowerCase();
    const isGemini = n.startsWith('gemini-');
    const isImage = (isGemini && n.includes('image')) || n.startsWith('image') || n.startsWith('imagen');
    if (isImage) return n.includes('flash') ? 'gemini-flash-image' : 'gemini-pro-image';
    if (isGemini && n.includes('flash')) return 'gemini-flash';
    if (isGemini && n.includes('pro')) return 'gemini-pro';
    // Keep Sonnet and Opus in separate quota buckets. Generic Claude/Haiku
    // names remain in the legacy category for old configs and aliases.
    if (n.includes('opus')) return 'claude-opus';
    if (n.includes('sonnet')) return 'claude-sonnet';
    if (n.includes('claude') || n.includes('haiku')) return 'claude';
    return 'other';
}

export interface ModelDisplayNameInput {
    name: string;
    display_name?: string;
}

export function getModelDisplayName(
    model: ModelDisplayNameInput | null | undefined,
    fallback?: string,
): string {
    if (model) {
        if (model.display_name) return model.display_name;
        if (model.name) return model.name;
    }
    return fallback ?? '';
}

/**
 * 按优先级查找配额模型：先精确匹配首选名，再按类别 fallback。
 */
export function findQuotaModel<T extends { name: string }>(
    models: T[] | undefined,
    category: ModelCategory,
): T | undefined {
    if (!models || models.length === 0) return undefined;
    const preferred: Partial<Record<ModelCategory, string[]>> = {
        'gemini-pro': ['gemini-pro-agent', 'gemini-3.1-pro-high', 'gemini-3.1-pro', 'gemini-3.1-pro-low', 'gemini-2.5-pro'],
        'gemini-flash': ['gemini-3-flash-agent', 'gemini-3-flash', 'gemini-3.5-flash'],
        'claude-sonnet': [CLAUDE_SONNET_QUOTA_MODEL, 'claude-sonnet-4-6-thinking'],
        'claude-opus': [CLAUDE_OPUS_QUOTA_MODEL, 'claude-opus-4-6'],
        'claude': [CLAUDE_SONNET_QUOTA_MODEL, CLAUDE_OPUS_QUOTA_MODEL],
    };
    const names = preferred[category];
    if (names) {
        for (const name of names) {
            const found = models.find(m => m.name === name);
            if (found) return found;
        }
    }
    return models.find(m => categorizeModel(m.name) === category);
}

export function getModelProtectionKey(name: string): string | null {
    switch (categorizeModel(name)) {
        case 'gemini-flash': return 'gemini-3-flash';
        case 'gemini-pro': return 'gemini-3-pro-high';
        case 'gemini-flash-image': return 'gemini-3.1-flash-image';
        case 'gemini-pro-image': return 'gemini-3-pro-image';
        case 'claude-sonnet': return CLAUDE_SONNET_QUOTA_MODEL;
        case 'claude-opus': return CLAUDE_OPUS_QUOTA_MODEL;
        case 'claude': return LEGACY_CLAUDE_QUOTA_MODEL;
        default: return null;
    }
}

export function isSupportedClaudeQuotaModel(name: string): boolean {
    const normalized = name.trim().toLowerCase();
    return (SUPPORTED_CLAUDE_QUOTA_MODELS as readonly string[]).includes(normalized);
}

/**
 * Check a model against account protection markers while retaining support for
 * old account files that stored one `claude` marker for both Claude buckets.
 */
export function isQuotaModelProtected(
    protectedModels: readonly string[] | undefined,
    modelName: string,
): boolean {
    if (!protectedModels || protectedModels.length === 0) return false;

    const normalizedModel = modelName.trim().toLowerCase();
    const targetKey = getModelProtectionKey(normalizedModel) ?? normalizedModel;
    const targetCategory = categorizeModel(normalizedModel);

    return protectedModels.some((marker) => {
        const normalizedMarker = marker.trim().toLowerCase();
        if (normalizedMarker === targetKey || normalizedMarker === normalizedModel) return true;

        // Legacy generic marker is intentionally broad for the two supported
        // Claude buckets, but does not make unrelated models protected.
        if (
            normalizedMarker === LEGACY_CLAUDE_QUOTA_MODEL
            && (targetCategory === 'claude-sonnet' || targetCategory === 'claude-opus')
        ) {
            return true;
        }

        const markerKey = getModelProtectionKey(normalizedMarker);
        return markerKey !== null && markerKey === targetKey;
    });
}

/**
 * 在任意图片类别中查找第一个实际模型。
 * 用于让新旧 image selector 共享同一配额槽位。
 */
export function findImageQuotaModel<T extends { name: string }>(
    models: T[] | undefined,
): T | undefined {
    if (!models || models.length === 0) return undefined;
    return models.find(m => {
        const c = categorizeModel(m.name);
        return c === 'gemini-flash-image' || c === 'gemini-pro-image';
    });
}

/** 账号管理 pin 列表缺省图像选择器时补入代表 Image，与仪表盘对齐。 */
export const DEFAULT_IMAGE_PIN_SELECTOR = 'gemini-3.1-flash-image';

export function ensurePinnedImageSelector(selectorIds: string[] | undefined): string[] {
    const pinned = selectorIds ? [...selectorIds] : [];
    const hasImage = pinned.some(id => {
        const category = categorizeModel(id);
        return category === 'gemini-flash-image' || category === 'gemini-pro-image';
    });
    if (hasImage) return pinned;
    pinned.push(DEFAULT_IMAGE_PIN_SELECTOR);
    return pinned;
}

export interface QuotaModelSelection<T> {
    selectorId: string;
    selectionKey: string;
    model: T | undefined;
}

export function resolveQuotaModels<T extends { name: string }>(
    models: T[] | undefined,
    selectorIds: string[],
): QuotaModelSelection<T>[] {
    const seen = new Set<string>();
    const results: QuotaModelSelection<T>[] = [];

    // Older settings used one `claude` selector. Expand it into two physical
    // selectors so one depleted bucket never masks the other.
    const expandedSelectorIds = selectorIds.flatMap((selectorId) => {
        const normalized = selectorId.trim().toLowerCase();
        return normalized === LEGACY_CLAUDE_QUOTA_MODEL
            ? [...SUPPORTED_CLAUDE_QUOTA_MODELS]
            : [selectorId];
    });

    for (const selectorId of expandedSelectorIds) {
        const normalizedId = selectorId.trim().toLowerCase();
        const category = categorizeModel(normalizedId);

        const isImage = category === 'gemini-pro-image' || category === 'gemini-flash-image';
        const selectionKey = isImage
            ? 'category:gemini-image'
            : category === 'other'
                ? `model:${normalizedId}`
                : `category:${category}`;

        if (seen.has(selectionKey)) continue;
        seen.add(selectionKey);

        const model = isImage
            ? findImageQuotaModel(models)
            : category === 'other' || category === 'claude'
                ? models?.find(m => m.name.trim().toLowerCase() === normalizedId)
                : findQuotaModel(models, category);

        results.push({ selectorId, selectionKey, model });
    }
    return results;
}
