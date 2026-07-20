import { TrendingUp } from 'lucide-react';
import { Account } from '../../types/account';
import { findQuotaModel } from '../../config/modelConfig';

interface BestAccountsProps {
    accounts: Account[];
    currentAccountId?: string;
    onSwitch?: (accountId: string) => void;
}

import { useTranslation } from 'react-i18next';

function BestAccounts({ accounts, currentAccountId, onSwitch }: BestAccountsProps) {
    const { t } = useTranslation();
    // 1. 获取按配额排序的列表 (排除当前账号)
    const geminiSorted = accounts
        .filter(a => a.id !== currentAccountId)
        .map(a => {
            const proQuota = findQuotaModel(a.quota?.models, 'gemini-pro')?.percentage || 0;
            const flashQuota = findQuotaModel(a.quota?.models, 'gemini-flash')?.percentage || 0;
            // 综合评分：Pro 权重更高 (70%)，Flash 权重 30%
            return {
                ...a,
                quotaVal: Math.round(proQuota * 0.7 + flashQuota * 0.3),
            };
        })
        .filter(a => a.quotaVal > 0)
        .sort((a, b) => b.quotaVal - a.quotaVal);

    const claudeSonnetSorted = accounts
        .filter(a => a.id !== currentAccountId)
        .map(a => ({
            ...a,
            quotaVal: findQuotaModel(a.quota?.models, 'claude-sonnet')?.percentage || 0,
        }))
        .filter(a => a.quotaVal > 0)
        .sort((a, b) => b.quotaVal - a.quotaVal);

    const claudeOpusSorted = accounts
        .filter(a => a.id !== currentAccountId)
        .map(a => ({
            ...a,
            quotaVal: findQuotaModel(a.quota?.models, 'claude-opus')?.percentage || 0,
        }))
        .filter(a => a.quotaVal > 0)
        .sort((a, b) => b.quotaVal - a.quotaVal);

    let bestGemini = geminiSorted[0];
    const bestClaudeSonnet = claudeSonnetSorted[0];
    const bestClaudeOpus = claudeOpusSorted[0];

    // 构造最终用于显示的视图模型 (兼容原有渲染逻辑)
    const bestGeminiRender = bestGemini ? { ...bestGemini, geminiQuota: bestGemini.quotaVal } : undefined;
    const bestClaudeSonnetRender = bestClaudeSonnet
        ? { ...bestClaudeSonnet, claudeQuota: bestClaudeSonnet.quotaVal }
        : undefined;
    const bestClaudeOpusRender = bestClaudeOpus
        ? { ...bestClaudeOpus, claudeQuota: bestClaudeOpus.quotaVal }
        : undefined;

    return (
        <div className="bg-white dark:bg-base-100 rounded-xl p-4 shadow-sm border border-gray-100 dark:border-base-200 h-full flex flex-col">
            <h2 className="text-base font-semibold text-gray-900 dark:text-base-content mb-3 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-blue-500 dark:text-blue-400" />
                {t('dashboard.best_accounts')}
            </h2>

            <div className="space-y-2 flex-1">
                {/* Gemini 最佳 */}
                {bestGeminiRender && (
                    <div className="flex items-center justify-between p-2.5 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-100 dark:border-green-900/30">
                        <div className="flex-1 min-w-0">
                            <div className="text-[10px] text-green-600 dark:text-green-400 font-medium mb-0.5">{t('dashboard.for_gemini')}</div>
                            <div className="font-medium text-sm text-gray-900 dark:text-base-content truncate">
                                {bestGeminiRender.email}
                            </div>
                        </div>
                        <div className="ml-2 px-2 py-0.5 bg-green-500 text-white text-xs font-semibold rounded-full">
                            {bestGeminiRender.geminiQuota}%
                        </div>
                    </div>
                )}

                {/* Claude Sonnet 最佳 */}
                {bestClaudeSonnetRender && (
                    <div className="flex items-center justify-between p-2.5 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg border border-cyan-100 dark:border-cyan-900/30">
                        <div className="flex-1 min-w-0">
                            <div className="text-[10px] text-cyan-600 dark:text-cyan-400 font-medium mb-0.5">{t('dashboard.for_claude_sonnet', 'Claude Sonnet')}</div>
                            <div className="font-medium text-sm text-gray-900 dark:text-base-content truncate">
                                {bestClaudeSonnetRender.email}
                            </div>
                        </div>
                        <div className="ml-2 px-2 py-0.5 bg-cyan-500 text-white text-xs font-semibold rounded-full">
                            {bestClaudeSonnetRender.claudeQuota}%
                        </div>
                    </div>
                )}

                {/* Claude Opus 最佳 */}
                {bestClaudeOpusRender && (
                    <div className="flex items-center justify-between p-2.5 bg-teal-50 dark:bg-teal-900/20 rounded-lg border border-teal-100 dark:border-teal-900/30">
                        <div className="flex-1 min-w-0">
                            <div className="text-[10px] text-teal-600 dark:text-teal-400 font-medium mb-0.5">{t('dashboard.for_claude_opus', 'Claude Opus')}</div>
                            <div className="font-medium text-sm text-gray-900 dark:text-base-content truncate">
                                {bestClaudeOpusRender.email}
                            </div>
                        </div>
                        <div className="ml-2 px-2 py-0.5 bg-teal-500 text-white text-xs font-semibold rounded-full">
                            {bestClaudeOpusRender.claudeQuota}%
                        </div>
                    </div>
                )}

                {(!bestGeminiRender && !bestClaudeSonnetRender && !bestClaudeOpusRender) && (
                    <div className="text-center py-4 text-gray-400 text-sm">
                        {t('accounts.no_data')}
                    </div>
                )}
            </div>

            {(bestGeminiRender || bestClaudeSonnetRender || bestClaudeOpusRender) && onSwitch && (
                <div className="mt-auto pt-3">
                    <button
                        className="w-full px-3 py-1.5 bg-blue-500 text-white text-xs font-medium rounded-lg hover:bg-blue-600 transition-colors"
                        onClick={() => {
                            // 优先切换到配额更高的账号
                            const candidates = [
                                bestGeminiRender ? { id: bestGeminiRender.id, quota: bestGeminiRender.geminiQuota } : undefined,
                                bestClaudeSonnetRender ? { id: bestClaudeSonnetRender.id, quota: bestClaudeSonnetRender.claudeQuota } : undefined,
                                bestClaudeOpusRender ? { id: bestClaudeOpusRender.id, quota: bestClaudeOpusRender.claudeQuota } : undefined,
                            ].filter((candidate): candidate is { id: string; quota: number } => Boolean(candidate));
                            const targetId = candidates.sort((a, b) => b.quota - a.quota)[0]?.id;

                            if (onSwitch && targetId) {
                                onSwitch(targetId);
                            }
                        }}
                    >
                        {t('dashboard.switch_best')}
                    </button>
                </div>
            )}
        </div>
    );

}

export default BestAccounts;
